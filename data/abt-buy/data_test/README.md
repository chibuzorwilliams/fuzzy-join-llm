Dataset Transformations 

This folder contains transformed versions of the Abt–Buy product-matching dataset.
Each version lets us test how different matching methods behave when the text is distorted.


Files included

abt_ciphered_letters.csv

abt_ciphered_words.csv

abt_scrambled.csv

buy_ciphered_letters.csv

buy_ciphered_words.csv

buy_scrambled.csv

(originals are in the main folder)

Transformations Explained (Very Simple)
Ciphered Letters

Letters are replaced with other letters (A→Z, B→R, etc.)
→ Structure preserved, characters changed.
Example: Sony Camera → Yjlr Wxvlzi

Ciphered Words

Whole words are replaced with random words, consistently.
→ Word positions kept, meaning removed.
Example: Sony Camera → Lamp Orange

Scrambled

Each word’s characters are shuffled.
→ Simulates typos/noise.
Example: Sony Camera → nSoy aaCrem


Why these matter
These transformations help evaluate:

String methods (Jaro-Winkler, Levenshtein, etc.)

Embedding methods (SentenceTransformer, OpenAI Embeddings)

LLMs (GPT-4o-mini, etc.)

They show which models survive noise, typos, and meaningless text.

Summary Table
File Type	        What Changes	             Purpose
Ciphered Letters	Letters swapped	             Tests structure based matching
Ciphered Words	        Words replaced	             Tests context free matching
Scrambled	        Characters shuffled	     Tests typo robustness
