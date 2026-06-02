
import sys
import os

# Add the app directory to path so we can import the pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.rag_pipeline import SemanticRetriever

def test_chunking_logic():
    print("--- Testing Chunking Logic (Fast) ---")
    # Initialize without loading model if possible, or just test the method
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    
    # Create a dummy long document (120 words)
    words = [f"word{i}" for i in range(1, 121)]
    long_text = " ".join(words)
    
    # Test the chunking method directly
    print(f"Splitting 120 words with chunk_size=50 and overlap=10...")
    chunks = retriever._chunk_text(long_text, chunk_size=50, overlap=10)
    
    print(f"Total chunks created: {len(chunks)}")
    
    for i, content in enumerate(chunks):
        chunk_words = content.split()
        print(f"\nChunk {i} (Length: {len(chunk_words)} words):")
        print(f"First 5 words: {' '.join(chunk_words[:5])}")
        print(f"Last 5 words: {' '.join(chunk_words[-5:])}")

    # Verify overlap:
    # Chunk 0: word1 to word50
    # Chunk 1: starts at (50-10) = 40. So word41.
    if len(chunks) > 1:
        chunk0_last = chunks[0].split()[-1]
        chunk1_first = chunks[1].split()[0]
        print(f"\nOverlap Check:")
        print(f"Chunk 0 ends with: {chunk0_last}")
        print(f"Chunk 1 starts with: {chunk1_first}")
        
        if chunk1_first == "word41":
            print("Success: Overlap is exactly 10 words (word41 to word50)!")
        else:
            print(f"Note: Expected word41, got {chunk1_first}")

if __name__ == "__main__":
    test_chunking_logic()
