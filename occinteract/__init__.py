"""OccInteract shared data library.

Provides the 4D-occupancy dataset loading, processing and visualization
utilities shared by the data engines, the Occ-VAE and the OccDirector model.

Structure:
    core/        base handlers for batch (VLM) processing tasks
    processing/  parallel orchestration with checkpointing
    tools/       occupancy renderers
"""
