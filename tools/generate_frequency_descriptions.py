"""Dataset-neutral entry point for the compatible LLM description generator."""
try:
    from tools.generate_cub200_frequency_descriptions import main
except ImportError:
    from generate_cub200_frequency_descriptions import main

if __name__ == '__main__':
    raise SystemExit(main())
