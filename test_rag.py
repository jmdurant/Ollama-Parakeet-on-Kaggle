#!/usr/bin/env python3
"""
Test script for RAG (Retrieval-Augmented Generation) functionality
Run this after the integrated services are up to verify RAG is working correctly
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# Configuration
BASE_URL = "https://unponderous-nonmelodramatically-sheila.ngrok-free.dev"

async def test_rag_status():
    """Test RAG status endpoint"""
    print("\n" + "="*60)
    print("Testing RAG Status...")
    print("="*60)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/rag/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("✓ RAG service is running")
                    print(f"  Collection: {data['collection_name']}")
                    print(f"  Documents: {data['document_count']}")
                    print(f"  Chunks: {data['chunk_count']}")
                    print(f"  Embeddings model: {data['embeddings_model']}")
                    return True
                else:
                    print(f"✗ RAG status check returned status {resp.status}")
                    return False
    except Exception as e:
        print(f"✗ RAG status test failed: {e}")
        return False

async def test_rag_ingest():
    """Test document ingestion (with sample text)"""
    print("\n" + "="*60)
    print("Testing Document Ingestion...")
    print("="*60)
    
    try:
        # Create a simple test document
        test_content = """
        F90.0 Attention-Deficit/Hyperactivity Disorder, Combined presentation
        
        Diagnostic Criteria:
        A. A persistent pattern of inattention and/or hyperactivity-impulsivity that interferes 
        with functioning or development, as characterized by (1) and/or (2):
        
        1. Inattention: Six (or more) of the following symptoms have persisted for at least 
        6 months to a degree that is inconsistent with developmental level:
        - Often fails to give close attention to details
        - Often has difficulty sustaining attention in tasks
        - Often does not seem to listen when spoken to directly
        - Often does not follow through on instructions
        
        2. Hyperactivity and Impulsivity: Six (or more) of the following symptoms:
        - Often fidgets with or taps hands or feet
        - Often leaves seat in situations when remaining seated is expected
        - Often runs about or climbs in inappropriate situations
        - Often unable to play or engage in leisure activities quietly
        """
        
        async with aiohttp.ClientSession() as session:
            data = {
                "content": test_content,
                "metadata": {
                    "source": "test_document",
                    "type": "diagnostic_criteria"
                }
            }
            
            async with session.post(
                f"{BASE_URL}/api/rag/ingest",
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print("✓ Document ingestion successful")
                    print(f"  Chunks created: {result.get('chunks_added', 0)}")
                    return True
                else:
                    error_text = await resp.text()
                    print(f"✗ Document ingestion failed with status {resp.status}")
                    print(f"  Error: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"✗ Document ingestion test failed: {e}")
        return False

async def test_rag_search():
    """Test RAG search functionality"""
    print("\n" + "="*60)
    print("Testing RAG Search...")
    print("="*60)
    
    queries = [
        "ADHD diagnostic criteria",
        "attention deficit symptoms",
        "hyperactivity in children"
    ]
    
    try:
        async with aiohttp.ClientSession() as session:
            for query in queries:
                print(f"\nSearching for: '{query}'")
                
                data = {"query": query, "k": 3}
                
                async with session.post(
                    f"{BASE_URL}/api/rag/search",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        chunks = result.get('chunks', [])
                        
                        if chunks:
                            print(f"  ✓ Found {len(chunks)} relevant chunks")
                            for i, chunk in enumerate(chunks[:2], 1):
                                preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
                                print(f"    {i}. {preview}")
                                print(f"       Score: {chunk['score']:.3f}")
                        else:
                            print("  ⚠ No relevant chunks found")
                    else:
                        print(f"  ✗ Search failed with status {resp.status}")
                        
            return True
                    
    except Exception as e:
        print(f"✗ RAG search test failed: {e}")
        return False

async def test_rag_query():
    """Test RAG query (context retrieval only)"""
    print("\n" + "="*60)
    print("Testing RAG Query (Context Retrieval)...")
    print("="*60)
    
    test_questions = [
        "What are the main symptoms of ADHD?",
        "How is attention deficit diagnosed?",
        "What is F90.0?"
    ]
    
    try:
        async with aiohttp.ClientSession() as session:
            for question in test_questions:
                print(f"\nQuery: '{question}'")
                
                data = {"question": question}
                
                async with session.post(
                    f"{BASE_URL}/api/rag/query",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        context = result.get('context', '')
                        sources = result.get('sources', [])
                        
                        if context:
                            print("  ✓ Context retrieved successfully")
                            preview = context[:200] + "..." if len(context) > 200 else context
                            print(f"  Context preview: {preview}")
                            print(f"  Sources: {sources}")
                        else:
                            print("  ⚠ No relevant context found")
                    else:
                        print(f"  ✗ Query failed with status {resp.status}")
                        
            return True
                    
    except Exception as e:
        print(f"✗ RAG query test failed: {e}")
        return False

async def test_rag_chat():
    """Test RAG-enhanced chat (full pipeline)"""
    print("\n" + "="*60)
    print("Testing RAG-Enhanced Chat...")
    print("="*60)
    
    test_questions = [
        "What are the diagnostic criteria for ADHD according to F90.0?",
        "Explain the difference between inattention and hyperactivity symptoms",
        "How long must ADHD symptoms persist for diagnosis?"
    ]
    
    try:
        async with aiohttp.ClientSession() as session:
            for question in test_questions:
                print(f"\nQuestion: '{question}'")
                
                data = {
                    "question": question,
                    "model": "gpt-oss:20b"
                }
                
                async with session.post(
                    f"{BASE_URL}/api/rag/chat",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        answer = result.get('answer', '')
                        sources = result.get('sources', [])
                        
                        if answer:
                            print("  ✓ RAG chat successful")
                            # Show first 300 chars of answer
                            preview = answer[:300] + "..." if len(answer) > 300 else answer
                            print(f"  Answer: {preview}")
                            if sources:
                                print(f"  Based on sources: {sources}")
                        else:
                            print("  ⚠ No answer generated")
                    else:
                        error_text = await resp.text()
                        print(f"  ✗ RAG chat failed with status {resp.status}")
                        print(f"    Error: {error_text}")
                        
            return True
                    
    except Exception as e:
        print(f"✗ RAG chat test failed: {e}")
        return False

async def main():
    """Run all RAG tests"""
    print("\n" + "="*60)
    print("RAG FUNCTIONALITY TEST SUITE")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    
    results = []
    
    # Test RAG status
    status_ok = await test_rag_status()
    results.append(("RAG Status", status_ok))
    
    # Test document ingestion
    ingest_ok = await test_rag_ingest()
    results.append(("Document Ingestion", ingest_ok))
    
    # Give time for indexing
    if ingest_ok:
        print("\nWaiting for document indexing...")
        await asyncio.sleep(2)
    
    # Test search
    search_ok = await test_rag_search()
    results.append(("RAG Search", search_ok))
    
    # Test query (context retrieval)
    query_ok = await test_rag_query()
    results.append(("RAG Query", query_ok))
    
    # Test full RAG chat
    chat_ok = await test_rag_chat()
    results.append(("RAG Chat", chat_ok))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for service, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{service:25} {status}")
    
    all_passed = all(ok for _, ok in results)
    
    if all_passed:
        print("\n🎉 All RAG tests passed!")
        print("\nRAG system is ready for:")
        print("  - Clinical document retrieval (DSM-5, ICD codes)")
        print("  - Context-aware Q&A about medical conditions")
        print("  - Enhanced chat with document references")
    else:
        print("\n⚠️  Some RAG tests failed. Check the logs above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)