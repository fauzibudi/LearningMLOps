import logging

def setup_logger():
    """Configure logging."""
    logger = logging.getLogger('HousePricePrediction')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler('house_price_prediction.log')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger