#!/usr/bin/env python3
"""
NLTK Setup Script for Production Deployment
Downloads required NLTK data during build process
"""
import nltk
import ssl
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Download required NLTK data for production"""
    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        nltk_downloads = [
            'punkt',
            'stopwords', 
            'vader_lexicon',
            'wordnet',
            'omw-1.4',
            'averaged_perceptron_tagger'
        ]
        
        for package in nltk_downloads:
            try:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)
                logger.info(f"✅ Successfully downloaded: {package}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to download {package}: {e}")
                
        logger.info("🎉 NLTK setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ NLTK setup failed: {e}")
        return False

if __name__ == "__main__":
    setup_nltk() 