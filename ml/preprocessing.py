import os
import logging
import pandas as pd
import numpy as np
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
INTENT_TO_CATEGORY = {
    # Billing
    'check_payment_methods': 'Billing', 'payment_issue': 'Billing', 
    'check_invoice': 'Billing', 'get_invoice': 'Billing', 'check_cancellation_fee': 'Billing',
    # Delivery
    'delivery_options': 'Delivery', 'delivery_period': 'Delivery', 'track_order': 'Delivery',
    'change_shipping_address': 'Delivery', 'set_up_shipping_address': 'Delivery', 'cancel_order': 'Delivery',
    # Refund
    'check_refund_policy': 'Refund', 'get_refund': 'Refund', 'track_refund': 'Refund',
    # Account Access
    'create_account': 'Account Access', 'delete_account': 'Account Access', 
    'edit_account': 'Account Access', 'recover_password': 'Account Access', 
    'registration_problems': 'Account Access', 'switch_account': 'Account Access', 'newsletter_subscription': 'Account Access',
    # Technical Issue / General
    'change_order': 'Technical Issue', 'place_order': 'Technical Issue', 
    'complaint': 'Technical Issue', 'review': 'Technical Issue', 
    'contact_customer_service': 'Technical Issue', 'contact_human_agent': 'Technical Issue'
}

def annotate_sentiment(text: str) -> str:
    text_lower = text.lower()
    negative_words = ['not', 'never', 'hate', 'terrible', 'issue', 'problem', 'worst', 'angry', 'furious']
    positive_words = ['great', 'excellent', 'thank', 'happy', 'love', 'good']
    
    if any(word in text_lower for word in negative_words):
        return 'Negative'
    elif any(word in text_lower for word in positive_words):
        return 'Positive'
    return 'Neutral'

def annotate_priority(text: str, category: str) -> str:
    text_lower = text.lower()
    urgent_keywords = ['urgent', 'immediately', 'asap', 'now', 'furious', 'escalate']
    
    if any(word in text_lower for word in urgent_keywords) or category in ['Refund', 'Technical Issue']:
        return 'High'
    return 'Medium'

def prepare_dataset(output_path: str = "ml/data/cleaned_tickets.csv"):
    logging.info("Downloading Bitext dataset")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    df = dataset.to_pandas()
    
    logging.info(f"Raw dataset shape: {df.shape}")

    df['category'] = df['intent'].map(INTENT_TO_CATEGORY)
    df = df.dropna(subset=['category']) 
    df = df[['instruction', 'category', 'response']].rename(
        columns={'instruction': 'text', 'response': 'agent_response'}
    )
    logging.info("Applying heuristic annotations")
    df['sentiment'] = df['text'].apply(annotate_sentiment)
    df['priority'] = df.apply(lambda row: annotate_priority(row['text'], row['category']), axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logging.info(f"dataset saved to {output_path}")
    logging.info(f"Final shape: {df.shape}")
    logging.info(f"Category Distribution:\n{df['category'].value_counts()}")
    
    return df

if __name__ == "__main__":
    prepare_dataset()