"""
Data Augmentation for Vietnamese Emotion Classification using LLMs with LangChain

This module provides tools for augmenting Vietnamese emotion classification datasets
using Large Language Models (LLMs) integrated with LangChain.

Supported provider:
- Google Gemini (Free tier available)

Author: Generated for Vietnamese Emotion Classification Research
Date: November 2025
"""

import os
import pandas as pd
from typing import List, Dict, Optional
import time
from tqdm import tqdm

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VietnameseEmotionAugmenter:
    """
    Data augmentation for Vietnamese emotion classification using LLMs
    
    This class provides methods to:
    1. Generate synthetic training samples
    2. Balance imbalanced datasets
    3. Increase dataset diversity
    
    Example:
        >>> augmenter = VietnameseEmotionAugmenter(model_provider='google')
        >>> augmented_df = augmenter.augment_dataset(df_train, num_variations_per_sample=2)
    """
    
    EMOTIONS = {
        'Enjoyment': 'vui vẻ',
        'Disgust': 'khó chịu',
        'Other': 'khác',
        'Sadness': 'buồn bã',
        'Anger': 'giận dữ',
        'Fear': 'sợ hãi',
        'Surprise': 'ngạc nhiên'
    }
    
    def __init__(
        self, 
        model_provider: str = 'google', 
        model_name: Optional[str] = None, 
        temperature: float = 0.8, 
        verbose: bool = True
    ):
        """
        Initialize the augmenter with Google Gemini LLM
        
        Args:
            model_provider: 'google' (only Google Gemini supported)
            model_name: Specific model name. Options:
                - 'gemini-1.5-flash' (default, fast and free)
                - 'gemini-1.5-pro' (higher quality, still free tier)
                - 'gemini-2.0-flash-exp' (experimental, latest)
            temperature: Creativity level (0.7-1.0 recommended for diversity)
            verbose: Print progress messages
        """
        self.temperature = temperature
        self.model_provider = model_provider
        self.verbose = verbose
        
        # Initialize Google Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        if self.verbose:
            print(f"✅ Initialized Google Gemini ({model_name or 'gemini-1.5-flash'})")
        
        # Create augmentation prompt template
        self.augmentation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một chuyên gia về ngôn ngữ tiếng Việt và phân tích cảm xúc. 
Nhiệm vụ của bạn là tạo ra các câu tiếng Việt MỚI thể hiện cảm xúc được chỉ định.
Các câu phải tự nhiên, như người Việt thường nói trên mạng xã hội."""),
            ("human", """Tạo {num_variations} câu tiếng Việt MỚI thể hiện cảm xúc "{emotion}" ({emotion_vn}).

Câu ví dụ: "{original_text}"

YÊU CẦU:
1. Tạo {num_variations} câu HOÀN TOÀN KHÁC NHAU (không phải paraphrase câu gốc)
2. Mỗi câu phải thể hiện RÕ RÀNG cảm xúc {emotion_vn}
3. Sử dụng ngôn ngữ tự nhiên, thông tục như người Việt dùng trên mạng
4. Có thể dùng emoji phù hợp (😊, 😡, 😢, 😱, 😤, etc.)
5. Độ dài câu đa dạng: ngắn (5-10 từ), trung bình (10-20 từ), dài (20+ từ)
6. Chỉ trả về các câu, mỗi câu một dòng, KHÔNG giải thích

Các câu mới:""")
        ])
        
        # Create chain
        self.augmentation_chain = self.augmentation_prompt | self.llm
    
    def augment_single(
        self, 
        emotion: str, 
        original_text: str, 
        num_variations: int = 3
    ) -> List[str]:
        """
        Generate variations for a single text
        
        Args:
            emotion: Emotion label in English (e.g., 'Enjoyment', 'Anger')
            original_text: Original Vietnamese text
            num_variations: Number of variations to generate
            
        Returns:
            List of generated Vietnamese texts
        """
        emotion_vn = self.EMOTIONS.get(emotion, emotion)
        
        try:
            # Invoke the chain
            result = self.augmentation_chain.invoke({
                "emotion": emotion,
                "emotion_vn": emotion_vn,
                "original_text": original_text,
                "num_variations": num_variations
            })
            
            # Extract content
            result_text = result.content
            
            # Parse the result
            variations = [line.strip() for line in result_text.strip().split('\n') if line.strip()]
            
            # Remove numbering if present (1., 2., etc.)
            cleaned_variations = []
            for line in variations:
                if line and line[0].isdigit() and '. ' in line:
                    line = line.split('. ', 1)[-1]
                elif line and line[0].isdigit() and ') ' in line:
                    line = line.split(') ', 1)[-1]
                cleaned_variations.append(line.strip())
            
            return cleaned_variations[:num_variations]
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Error generating variations: {e}")
            return []
    
    def augment_dataset(
        self, 
        df: pd.DataFrame, 
        num_variations_per_sample: int = 2,
        max_samples_per_emotion: Optional[int] = None,
        emotions_to_augment: Optional[List[str]] = None,
        delay_seconds: float = 0.5
    ) -> pd.DataFrame:
        """
        Augment entire dataset
        
        Args:
            df: DataFrame with columns ['Emotion', 'Sentence']
            num_variations_per_sample: How many variations per original text
            max_samples_per_emotion: Limit samples per emotion (None for all)
            emotions_to_augment: List of emotions to augment (None for all)
            delay_seconds: Delay between API calls to avoid rate limits
            
        Returns:
            DataFrame with original + augmented data
        """
        augmented_rows = []
        
        # Determine which emotions to augment
        emotions = emotions_to_augment if emotions_to_augment else df['Emotion'].unique()
        
        # Process by emotion
        for emotion in emotions:
            emotion_df = df[df['Emotion'] == emotion]
            
            if max_samples_per_emotion:
                emotion_df = emotion_df.head(max_samples_per_emotion)
            
            if self.verbose:
                print(f"\n🔄 Augmenting {emotion}: {len(emotion_df)} samples...")
            
            for idx, row in tqdm(emotion_df.iterrows(), total=len(emotion_df), desc=emotion, disable=not self.verbose):
                variations = self.augment_single(
                    emotion=row['Emotion'],
                    original_text=row['Sentence'],
                    num_variations=num_variations_per_sample
                )
                
                # Add variations to results
                for var_text in variations:
                    if var_text:  # Only add non-empty variations
                        augmented_rows.append({
                            'Emotion': row['Emotion'],
                            'Sentence': var_text,
                            'emotion_vn': row.get('emotion_vn', self.EMOTIONS.get(row['Emotion'], '')),
                            'augmented': True,
                            'source': 'llm_generated'
                        })
                
                # Delay to avoid rate limits
                time.sleep(delay_seconds)
        
        # Create augmented dataframe
        augmented_df = pd.DataFrame(augmented_rows)
        
        # Mark original data
        df_original = df.copy()
        df_original['augmented'] = False
        df_original['source'] = 'original'
        
        # Combine
        combined_df = pd.concat([df_original, augmented_df], ignore_index=True)
        
        if self.verbose:
            print(f"\n✅ Augmentation complete!")
            print(f"   Original samples: {len(df)}")
            print(f"   Augmented samples: {len(augmented_df)}")
            print(f"   Total samples: {len(combined_df)}")
        
        return combined_df
    
    def balance_dataset(
        self, 
        df: pd.DataFrame, 
        target_samples_per_emotion: int,
        delay_seconds: float = 0.5
    ) -> pd.DataFrame:
        """
        Balance dataset by augmenting minority classes
        
        Args:
            df: Original DataFrame with columns ['Emotion', 'Sentence']
            target_samples_per_emotion: Target number of samples per emotion
            delay_seconds: Delay between API calls
            
        Returns:
            Balanced DataFrame
        """
        balanced_rows = []
        
        for emotion in df['Emotion'].unique():
            emotion_df = df[df['Emotion'] == emotion]
            current_count = len(emotion_df)
            
            # Add all original samples
            for _, row in emotion_df.iterrows():
                balanced_rows.append({
                    'Emotion': row['Emotion'],
                    'Sentence': row['Sentence'],
                    'emotion_vn': row.get('emotion_vn', self.EMOTIONS.get(row['Emotion'], '')),
                    'augmented': False,
                    'source': 'original'
                })
            
            # Augment if needed
            if current_count < target_samples_per_emotion:
                needed = target_samples_per_emotion - current_count
                
                if self.verbose:
                    print(f"\n🔄 {emotion}: {current_count} → {target_samples_per_emotion} (need {needed} more)")
                
                # Sample with replacement if needed
                samples_to_augment = emotion_df.sample(n=needed, replace=(needed > current_count))
                
                for idx, row in tqdm(samples_to_augment.iterrows(), total=len(samples_to_augment), desc=emotion, disable=not self.verbose):
                    variations = self.augment_single(
                        emotion=row['Emotion'],
                        original_text=row['Sentence'],
                        num_variations=1
                    )
                    
                    for var_text in variations:
                        if var_text:
                            balanced_rows.append({
                                'Emotion': row['Emotion'],
                                'Sentence': var_text,
                                'emotion_vn': row.get('emotion_vn', self.EMOTIONS.get(row['Emotion'], '')),
                                'augmented': True,
                                'source': 'llm_balanced'
                            })
                    
                    time.sleep(delay_seconds)
            else:
                if self.verbose:
                    print(f"✅ {emotion}: {current_count} (already sufficient)")
        
        result_df = pd.DataFrame(balanced_rows)
        
        if self.verbose:
            print(f"\n✅ Balancing complete!")
            print(f"\n📊 Final distribution:")
            print(result_df['Emotion'].value_counts())
        
        return result_df


# Convenience functions for common use cases

def augment_training_data(
    input_path: str, 
    output_path: str, 
    model_provider: str = 'google',
    model_name: str = 'gemini-1.5-flash',
    variations_per_sample: int = 2,
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to augment training data from a CSV file
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save augmented CSV file
        model_provider: 'google' (only supported provider)
        model_name: Gemini model name (default: 'gemini-1.5-flash')
        variations_per_sample: Number of variations per sample
        max_samples: Maximum samples per emotion (for testing)
        
    Returns:
        Augmented DataFrame
    """
    # Load data
    df = pd.read_csv(input_path)
    print(f"📊 Loaded {len(df)} samples from {input_path}")
    print(f"   Emotion distribution:\n{df['Emotion'].value_counts()}")
    
    # Initialize augmenter
    augmenter = VietnameseEmotionAugmenter(
        model_provider=model_provider,
        model_name=model_name,
        temperature=0.8
    )
    
    # Augment
    augmented_df = augmenter.augment_dataset(
        df=df,
        num_variations_per_sample=variations_per_sample,
        max_samples_per_emotion=max_samples,
        delay_seconds=0.5
    )
    
    # Save
    augmented_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n💾 Saved augmented data to {output_path}")
    
    return augmented_df


def balance_training_data(
    input_path: str, 
    output_path: str,
    target_per_emotion: int = 1000,
    model_provider: str = 'google',
    model_name: str = 'gemini-1.5-flash'
) -> pd.DataFrame:
    """
    Convenience function to balance dataset by augmenting minority classes
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save balanced CSV file
        target_per_emotion: Target number of samples per emotion
        model_provider: 'google' (only supported provider)
        model_name: Gemini model name (default: 'gemini-1.5-flash')
        
    Returns:
        Balanced DataFrame
    """
    # Load data
    df = pd.read_csv(input_path)
    print(f"📊 Original distribution:\n{df['Emotion'].value_counts()}")
    
    # Initialize augmenter
    augmenter = VietnameseEmotionAugmenter(
        model_provider=model_provider,
        model_name=model_name,
        temperature=0.8
    )
    
    # Balance
    balanced_df = augmenter.balance_dataset(df, target_per_emotion)
    
    # Save
    balanced_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Balanced distribution:\n{balanced_df['Emotion'].value_counts()}")
    print(f"💾 Saved to {output_path}")
    
    return balanced_df


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment Vietnamese emotion classification data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--mode', type=str, choices=['augment', 'balance'], default='augment',
                       help='Augmentation mode: augment all or balance classes')
    parser.add_argument('--provider', type=str, default='google',
                       help='LLM provider (only google supported)')
    parser.add_argument('--model', type=str, default='gemini-1.5-flash',
                       help='Gemini model name (gemini-1.5-flash, gemini-1.5-pro, etc.)')
    parser.add_argument('--variations', type=int, default=2,
                       help='Number of variations per sample (augment mode)')
    parser.add_argument('--target', type=int, default=1000,
                       help='Target samples per emotion (balance mode)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per emotion (for testing)')
    
    args = parser.parse_args()
    
    if args.mode == 'augment':
        augment_training_data(
            input_path=args.input,
            output_path=args.output,
            model_provider=args.provider,
            model_name=args.model,
            variations_per_sample=args.variations,
            max_samples=args.max_samples
        )
    else:
        balance_training_data(
            input_path=args.input,
            output_path=args.output,
            target_per_emotion=args.target,
            model_provider=args.provider,
            model_name=args.model
        )
