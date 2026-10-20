"""Generate genuine LLM scale descriptions for CUB, CIFAR-100 or mini-ImageNet."""
try:
    from tools.generate_cub200_frequency_descriptions_deepseek import main
except ImportError:
    from generate_cub200_frequency_descriptions_deepseek import main

if __name__ == '__main__':
    raise SystemExit(main())
