"""Shared, leakage-aware utilities for the CYBEX EEG paper cluster.

Each paper repository ships only the modules it uses. All functions that
split, normalise or select features document which partition they are fitted
on, because that is what decides whether a reported accuracy is leakage-free.
"""
__version__ = "1.0.0"
