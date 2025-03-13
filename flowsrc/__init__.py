

# import logging


# ### Set up package wide logging, if none is

# Check if the root logger has handlers configured
# if not logging.getLogger().hasHandlers():
    # Configure logging with both console and file handlers
#     logging.basicConfig(
#         level=logging.INFO,  # Set the logging level
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.StreamHandler(),  # Console handler
#             logging.FileHandler('log.log', mode='w')  # File handler with write mode
#         ]
#    )

# logging.captureWarnings(True)
# Define a package-level logger
# logger = logging.getLogger(__name__)
