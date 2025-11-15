# Dataset Transformations

This folder contains transformed versions of the **Abt–Buy** product matching dataset.  
Each version helps test how matching methods behave when product text is distorted.

---

##  Files Included
- `abt_ciphered_letters.csv`
- `abt_ciphered_words.csv`
- `abt_scrambled.csv`
- `buy_ciphered_letters.csv`
- `buy_ciphered_words.csv`
- `buy_scrambled.csv`

Original datasets remain unchanged in the main folder.

---

#  Transformations Explained

##  1. Ciphered Letters  
Each letter is replaced with another letter using a fixed substitution map  
(e.g., A→Z, B→R).

-> Structure preserved  
-> Word boundaries preserved  
-> Human meaning lost  

**Example:**  
`Sony Camera` → `Yjlr Wxvlzi`

---

##  2. Ciphered Words  
Entire words are replaced with consistent but random substitutes.

-> Sentence structure preserved  
-> No semantic meaning  

**Example:**  
`Sony Camera` → `Lamp Orange`

---

##  3. Scrambled  
Characters within each word are shuffled.

-> Same letters as original  
-> Length preserved  
-> Looks like extreme typos  

**Example:**  
`Sony Camera` → `nSoy aaCrem`

---

#  Why These Transformations Matter
They allow evaluation of:

### **String based methods**
- Jaro-Winkler  
- Levenshtein  
- Monge-Elkan  
- Soft TF‑IDF  
- TF‑IDF  

-> Useful for testing sensitivity to character level noise.

### **Embedding based methods**
- SentenceTransformer  
- OpenAI Embeddings  

-> Useful for testing semantic robustness.

### **LLM-based Matching**
- GPT‑4o‑mini (Top‑k search + reasoning)  

-> Useful for testing reasoning through heavily distorted text.

---

#  Quick Summary Table

| File Type        | What Changes        | Purpose                     |
|------------------|---------------------|-----------------------------|
| Ciphered Letters | Letters swapped     | Tests structural similarity |
| Ciphered Words   | Words swapped       | Tests semantic independence |
| Scrambled        | Characters shuffled | Tests typo robustness       |

---

