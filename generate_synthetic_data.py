"""
Wrapper for data_generator.py
Provides full backwards compatibility while aliasing the synthetic data generator.
"""
from data_generator import main

if __name__ == "__main__":
    main()
