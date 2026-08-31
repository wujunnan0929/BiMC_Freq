import copy
import hashlib
import json
import os
from pathlib import Path
from random import SystemRandom

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DatasetManager:

    def __init__(self, cfg):

        # Properties
        self.cfg = cfg

        # Dataset split setting
        self.root = cfg.DATASET.ROOT
        self.dataset_name    = cfg.DATASET.NAME
        self.num_init_cls    = cfg.DATASET.NUM_INIT_CLS
        self.num_inc_cls     = cfg.DATASET.NUM_INC_CLS
        self.num_base_shot   = cfg.DATASET.NUM_BASE_SHOT
        self.num_inc_shot    = cfg.DATASET.NUM_INC_SHOT
        
        # training setting of data
        self.num_workers     = cfg.DATALOADER.NUM_WORKERS
        self.train_batchsize_base = cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE
        self.train_batchsize_inc = cfg.DATALOADER.TRAIN.BATCH_SIZE_INC
        self.test_batchsize = cfg.DATALOADER.TEST.BATCH_SIZE

        # setup data
        self._setup_data(self.root, self.dataset_name)
        self.class_index_in_task = []
        self.class_index_in_task.append(np.arange(0, self.num_init_cls))
        for start in range(self.num_init_cls, self.num_total_classes, self.num_inc_cls):
            end = min(start + self.num_inc_cls, self.num_total_classes)
            self.class_index_in_task.append(np.arange(start, end))
        self.num_tasks = len(self.class_index_in_task)
        self.train_transform, self.test_transform = self._set_transform()
        self._initialize_support_protocol()



    def _setup_data(self, root, dataset_name):
        full_dataset = get_data_source(root, dataset_name)
        self.class_names = full_dataset.classes
        self.template = full_dataset.template
        self.train_data, self.train_targets = full_dataset.get_train_data()
        self.test_data, self.test_targets = full_dataset.get_test_data()

        # convert labels  to `np.ndarray` for convenient indexing
        if not isinstance(self.train_targets, np.ndarray):
            self.train_targets = np.array(self.train_targets)
        if not isinstance(self.test_targets, np.ndarray):
            self.test_targets = np.array(self.test_targets)
        
        self.num_total_classes = len(self.class_names)
    

    def get_dataset(self, task_id, source, mode=None, accumulated_past=False):
        '''
        source: which part of dataset
        mode: which data transform is used
        accumulated_past (Bool): Whether the training data in this contains the data from the past 
        '''
        assert 0 <= task_id < len(self.class_index_in_task), \
               f"task id {task_id} should be in range [0, {len(self.class_index_in_task) - 1}]"

        # Transform changes must never cause another support-set draw.
        self._ensure_support_protocol()

        # Get data
        if source == 'train':
            # When training, using data of task [i]
            x, y = self.train_data, self.train_targets
            if accumulated_past:
                class_idx = np.concatenate(self.class_index_in_task[0: task_id + 1])
            else:
                class_idx = self.class_index_in_task[task_id]
            if accumulated_past:
                sample_indices = np.concatenate(self._session_support_indices[:task_id + 1])
            else:
                sample_indices = self._session_support_indices[task_id]

        elif source == 'test':
            # When testing, using data of tasks [0..i]
            x, y = self.test_data, self.test_targets
            class_idx = np.concatenate(self.class_index_in_task[0: task_id + 1])
            sample_indices = self._indices_for_classes(y, class_idx)

        else:
            raise ValueError(f'Invalid data source :{source}')
        
        # Get Transform
        if mode == 'train':
            transform = self.train_transform
        elif mode == 'test':
            transform = self.test_transform
        else:
            raise ValueError(f'Invalid transform mode: {mode}')

        def find_sublist_indices(matrix, numbers):
            """
            Function to find the indices of the sublists where each number in 'numbers' is located.

            Parameters:
            matrix (list of list of int): The 2D list to search in.
            numbers (np.ndarray): The numpy array of numbers to search for.

            Returns:
            dict: A dictionary with keys as the numbers from 'numbers' and values as the indices of the sublists.
            """
            indices = {}
            for x in numbers:
                found = False
                for i, sublist in enumerate(matrix):
                    if x in sublist:
                        indices[x] = i
                        found = True
                        break
                if not found:
                    indices[x] = -1  # If number not found, set index to -1
            return indices
        
        class_to_task_id = find_sublist_indices(self.class_index_in_task, class_idx)
        data = np.asarray(x)[sample_indices]
        targets = np.asarray(y)[sample_indices]
        task_dataset = TaskDataset(
            data, targets, transform, class_to_task_id, self.class_names,
            sample_indices=sample_indices,
        )
        return task_dataset

    @staticmethod
    def _indices_for_classes(targets, class_ids, shot=None, rng=None):
        """Select dataset-global indices, keeping the established class ordering."""
        targets = np.asarray(targets)
        selected = []
        for class_id in class_ids:
            candidates = np.flatnonzero(targets == class_id)
            if shot is not None and shot != -1:
                if shot < 1:
                    raise ValueError(f"Support shot must be positive or -1, got {shot}")
                if shot > len(candidates):
                    print(f'shot:{shot} is greater than num of sample:{len(candidates)} in class{class_id}')
                else:
                    if rng is None:
                        raise ValueError("Few-shot selection requires an independent random generator")
                    candidates = rng.choice(candidates, size=shot, replace=False)
            selected.append(candidates)
        return np.concatenate(selected).astype(np.int64, copy=False)

    def _ensure_support_protocol(self):
        # Lazy initialization also supports lightweight managers used by tests/tools.
        if not hasattr(self, '_session_support_indices'):
            self._initialize_support_protocol()

    def _initialize_support_protocol(self):
        dataset_cfg = getattr(getattr(self, 'cfg', None), 'DATASET', None)
        requested_seed = int(getattr(dataset_cfg, 'SUPPORT_SEED', -1))
        self._explicit_support_seed = requested_seed
        if requested_seed < 0:
            requested_seed = int(getattr(getattr(self, 'cfg', None), 'SEED', -1))
        if requested_seed < 0:
            requested_seed = SystemRandom().randrange(2 ** 32)
        if requested_seed >= 2 ** 32:
            raise ValueError("Support seed must be in [0, 2**32 - 1]")
        self.support_seed = requested_seed
        self._support_rng = np.random.RandomState(self.support_seed)
        manifest_path = getattr(dataset_cfg, 'SUPPORT_MANIFEST', '')
        if manifest_path and Path(manifest_path).exists():
            self.load_support_manifest(manifest_path)
            return
        selections = []
        for task_id, class_ids in enumerate(self.class_index_in_task):
            shot = self.num_base_shot if task_id == 0 else self.num_inc_shot
            indices = self._indices_for_classes(
                self.train_targets, class_ids, shot=shot, rng=self._support_rng,
            )
            indices.setflags(write=False)
            selections.append(indices)
        self._session_support_indices = tuple(selections)
        if manifest_path:
            self.save_support_manifest(manifest_path)

    def _training_sample_fingerprint(self):
        """Hash the ordered sample identities without opening path-backed images.

        In-memory images include dtype, shape, and every byte in logical C order.
        Streaming bounds temporary copies even for non-contiguous image arrays.
        """
        values = np.asarray(self.train_data)
        digest = hashlib.sha256()
        if values.ndim == 1 and values.dtype.kind in ('U', 'S'):
            digest.update(b'ordered-paths-v1\0')
            for path in values:
                encoded = os.fsdecode(path).encode('utf-8', errors='surrogatepass')
                # Length prefixes avoid ambiguity when filenames contain separators.
                digest.update(len(encoded).to_bytes(8, 'little'))
                digest.update(encoded)
        else:
            if values.ndim < 1 or values.dtype.hasobject:
                raise TypeError('Training data must be an image ndarray or a 1-D sequence of paths')
            digest.update(b'ordered-ndarray-v1\0')
            digest.update(json.dumps(
                {'dtype': values.dtype.str, 'shape': list(values.shape)},
                sort_keys=True, separators=(',', ':'),
            ).encode('ascii'))
            chunk_bytes = 1024 * 1024
            if values.flags.c_contiguous:
                raw = memoryview(values).cast('B')
                for start in range(0, len(raw), chunk_bytes):
                    digest.update(raw[start:start + chunk_bytes])
            else:
                iterator = np.nditer(
                    values, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    op_flags=['readonly'], order='C',
                    buffersize=max(1, chunk_bytes // max(1, values.dtype.itemsize)),
                )
                for block in iterator:
                    digest.update(block.tobytes(order='C'))
        return digest.hexdigest()

    def _support_metadata(self, refresh=False):
        if refresh or not hasattr(self, '_support_metadata_cache'):
            targets = np.asarray(self.train_targets, dtype='<i8')
            if len(self.train_data) != len(targets):
                raise ValueError('Training sample and target counts do not match')
            self._support_metadata_cache = {
                'version': 2,
                'dataset': str(self.dataset_name).lower(),
                'train_size': len(targets),
                'train_targets_sha256': hashlib.sha256(targets.tobytes()).hexdigest(),
                'train_samples_sha256': self._training_sample_fingerprint(),
                'class_names': list(self.class_names),
                'class_groups': [np.asarray(group).astype(int).tolist()
                                 for group in self.class_index_in_task],
                'shots': {'base': self.num_base_shot, 'incremental': self.num_inc_shot},
            }
        # The manifest builder adds session fields; never let those mutate the cache.
        return copy.deepcopy(self._support_metadata_cache)

    def _support_manifest(self):
        manifest = self._support_metadata()
        manifest['seed'] = self.support_seed
        targets = np.asarray(self.train_targets)
        manifest['sessions'] = [
            {'task_id': task_id, 'indices': indices.tolist(),
             'targets': targets[indices].astype(int).tolist()}
            for task_id, indices in enumerate(self._session_support_indices)
        ]
        return manifest

    def save_support_manifest(self, path):
        """Export the fixed protocol; never silently replace a different one."""
        self._ensure_support_protocol()
        path = Path(path)
        manifest = self._support_manifest()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('x', encoding='utf-8') as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
                handle.write('\n')
        except FileExistsError:
            with path.open('r', encoding='utf-8') as handle:
                existing = json.load(handle)
            if existing != manifest:
                raise ValueError(f"Refusing to overwrite a different support manifest: {path}")
        return str(path)

    def load_support_manifest(self, path):
        """Load and validate indices without opening images or resampling classes."""
        path = Path(path)
        try:
            with path.open('r', encoding='utf-8') as handle:
                manifest = json.load(handle)
            if not isinstance(manifest, dict):
                raise ValueError("expected a JSON object")
            if manifest.get('version') != 2:
                raise ValueError('unsupported version; version 2 with sample identity hashes is required')
            # Loading is an explicit integrity check. Recompute even when an
            # already-used manager is asked to load after its data was modified.
            for key, expected in self._support_metadata(refresh=True).items():
                if manifest.get(key) != expected:
                    raise ValueError(f"{key} does not match the current dataset/protocol")
            seed = manifest.get('seed')
            if type(seed) is not int or not 0 <= seed < 2 ** 32:
                raise ValueError("seed must be an integer in [0, 2**32 - 1]")
            explicit_seed = getattr(self, '_explicit_support_seed', -1)
            if explicit_seed >= 0 and seed != explicit_seed:
                raise ValueError("seed does not match DATASET.SUPPORT_SEED")
            sessions = manifest.get('sessions')
            if not isinstance(sessions, list) or len(sessions) != len(self.class_index_in_task):
                raise ValueError("sessions must contain exactly one entry per task")
            targets = np.asarray(self.train_targets)
            selections, all_indices = [], set()
            for task_id, (session, classes) in enumerate(zip(sessions, self.class_index_in_task)):
                if not isinstance(session, dict) or session.get('task_id') != task_id:
                    raise ValueError(f"invalid task_id for session {task_id}")
                raw_indices = session.get('indices')
                if not isinstance(raw_indices, list) or any(type(i) is not int for i in raw_indices):
                    raise ValueError(f"session {task_id}: indices must be an integer list")
                indices = np.asarray(raw_indices, dtype=np.int64)
                if np.any(indices < 0) or np.any(indices >= len(targets)):
                    raise ValueError(f"session {task_id}: index outside training dataset")
                if len(set(raw_indices)) != len(raw_indices) or all_indices.intersection(raw_indices):
                    raise ValueError(f"session {task_id}: duplicate support index")
                selected_targets = targets[indices].astype(int)
                raw_targets = session.get('targets')
                if (not isinstance(raw_targets, list)
                        or any(type(y) is not int for y in raw_targets)
                        or raw_targets != selected_targets.tolist()):
                    raise ValueError(f"session {task_id}: saved targets do not match indices")
                shot = self.num_base_shot if task_id == 0 else self.num_inc_shot
                expected_targets = []
                for class_id in classes:
                    available = int(np.sum(targets == class_id))
                    count = available if shot in (None, -1) else min(shot, available)
                    if count < 1:
                        raise ValueError(f"session {task_id}: no support for class {class_id}")
                    expected_targets.extend([int(class_id)] * count)
                if selected_targets.tolist() != expected_targets:
                    raise ValueError(f"session {task_id}: class membership, ordering, or shot count mismatch")
                indices.setflags(write=False)
                selections.append(indices)
                all_indices.update(raw_indices)
            self.support_seed = seed
            self._support_rng = np.random.RandomState(seed)
            self._session_support_indices = tuple(selections)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"Invalid support manifest {path}: {error}") from error
        return str(path)
    

    
    def get_dataloader(self, task_id, source, mode=None, accumulate_past=False):
        assert source in ['train', 'test'], f'data source must be in ["train", "test"], got {source}'
        # the default mode is same as source
        if mode == None:
            mode = source
        dataset = self.get_dataset(task_id, source, mode, accumulate_past)
        if source == 'train':
            if task_id == 0:
                batchsize = self.train_batchsize_base
            else:
                batchsize = self.train_batchsize_inc
            loader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False,
                                pin_memory=True)
        elif source == 'test':
            loader = DataLoader(dataset,
                                batch_size=self.test_batchsize,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False,
                                pin_memory=True)
        else:
            raise ValueError(f'Invalid data source: {source}')
        return loader
    


    def _select_data_from_class_index(self, x, y, class_idx, shot, source):
        # Compatibility helper. Public loaders use the fixed session cache above.
        self._ensure_support_protocol()
        indices = self._indices_for_classes(
            y, class_idx, shot=shot if source == 'train' else None,
            rng=self._support_rng,
        )
        return np.asarray(x)[indices], np.asarray(y)[indices]
    

    def _set_transform(self):
        img_size = 224
        MEAN = [0.48145466, 0.4578275, 0.40821073]
        STD  = [0.26862954, 0.26130258, 0.27577711]
        train_transform  = transforms.Compose([
            # transforms.RandomResizedCrop(img_size, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return train_transform, test_transform
    


class TaskDataset(Dataset):
    def __init__(self, images, labels, transform, class_to_task_id=None, class_name=None,
                 sample_indices=None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.transform = transform
        self.use_path = bool(len(images)) and isinstance(images[0], str)
        self.class_to_task_id = class_to_task_id
        self.class_name = class_name
        self.sample_indices = np.asarray(
            np.arange(len(images)) if sample_indices is None else sample_indices,
            dtype=np.int64,
        ).copy()
        if len(self.sample_indices) != len(images):
            raise ValueError("Sample index count must match dataset size")
        self.sample_indices.setflags(write=False)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        if self.use_path:
            image = self.transform(pil_loader(self.images[idx]))
        else:
            image = self.transform(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        
        if self.class_to_task_id is not None:
            task_id = self.class_to_task_id[label]
        else:
            task_id = -1
        
        if self.class_name is not None:
            cls_name = self.class_name[label]
        else:
            cls_name = ''
            
        ret = {
            'idx': idx, 
            'global_index': int(self.sample_indices[idx]),
            'sample_id': int(self.sample_indices[idx]),
            'image': image,
            'label': label,
            'cls_name': cls_name,
            'task_id' : task_id
        }
        return ret



def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# NEED MODIFY HERE IF YOU WANT TO ADD NEW DATASETS
def get_data_source(root, name):
    from .cifar100 import CIFAR100
    from .miniimagenet import MiniImagenet
    from .cub200 import CUB200
    source_dict = {
        'cifar100' : CIFAR100,
        'miniimagenet' : MiniImagenet,
        'cub200': CUB200,
    }
    return source_dict[name.lower()](root=root)
