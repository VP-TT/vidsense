# VidSense

Ask questions about a video and get answers that point to the exact moments.

VidSense transcribes the audio with Whisper, picks keyframes with CLIP and K-means,
aligns the keyframes to the transcript with dynamic time warping (DTW), and indexes the
aligned chunks in ChromaDB. A LangChain pipeline retrieves the relevant chunks and asks
an LLM (DeepSeek-R1 via Ollama, or an OpenAI model) to answer with timestamps that the
Streamlit player can jump to.

> Work in progress: the pipeline is being built module by module.

## Pipeline

1. **Input**: decode the video into sampled frames and an audio track (PyAV).
2. **Audio**: Whisper (`faster-whisper`) produces a transcript with timestamps.
3. **Vision**: CLIP ViT-B/32 embeds the frames; temporal K-means picks keyframes.
4. **Alignment**: banded DTW matches keyframes to transcript segments.
5. **Retrieval**: aligned chunks are embedded with MiniLM (384-d) and stored in ChromaDB.
6. **Answer**: LangChain sends the retrieved chunks to the LLM; answers cite `[mm:ss]` timestamps.
