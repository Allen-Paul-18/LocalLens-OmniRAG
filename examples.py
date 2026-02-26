"""
Example Usage Scripts for Multimodal RAG System
Demonstrates various use cases and integration patterns
"""

import os
from pathlib import Path
from multimodal_rag_system import MultimodalRAGSystem

# ============================================================================
# EXAMPLE 1: Basic Document Search
# ============================================================================

def example_basic_search():
    """Simple search across documents"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Document Search")
    print("="*60)
    
    # Initialize system
    rag = MultimodalRAGSystem(storage_dir="./demo_rag_data")
    
    # Create some sample documents
    sample_docs = {
        'python_guide.txt': '''
            Python is a high-level programming language known for its simplicity.
            It supports multiple programming paradigms including procedural, 
            object-oriented, and functional programming.
        ''',
        'ai_overview.txt': '''
            Artificial Intelligence (AI) is transforming industries worldwide.
            Machine learning, a subset of AI, enables computers to learn from data.
            Deep learning uses neural networks to solve complex problems.
        ''',
        'web_dev.txt': '''
            Web development involves creating websites and web applications.
            Frontend technologies include HTML, CSS, and JavaScript.
            Backend development uses frameworks like Django, Flask, and FastAPI.
        '''
    }
    
    # Write and ingest sample documents
    for filename, content in sample_docs.items():
        with open(filename, 'w') as f:
            f.write(content)
        rag.ingest_document(filename)
        os.remove(filename)  # Clean up
    
    # Search examples
    queries = [
        "What is Python?",
        "Tell me about machine learning",
        "What frontend technologies are mentioned?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = rag.search(query, top_k=2)
        
        if results:
            print(f"Best match: {results[0]['metadata']['source_file']}")
            print(f"Relevance: {results[0]['relevance']:.1%}")
            print(f"Content: {results[0]['content'][:100]}...")


# ============================================================================
# EXAMPLE 2: Cross-Modal Search (Text → Image)
# ============================================================================

def example_cross_modal_search():
    """Search for images using text queries"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Cross-Modal Search")
    print("="*60)
    
    rag = MultimodalRAGSystem()
    
    # Note: This requires actual image files
    # In a real scenario, you would have images with relevant content
    
    print("\nTo run this example:")
    print("1. Place images in ./images/ directory")
    print("2. Ingest them: rag.ingest_document('images/chart.png')")
    print("3. Search: rag.search('financial chart', cross_modal=True)")
    print("\nThe system will find images that match your text query!")


# ============================================================================
# EXAMPLE 3: Meeting Intelligence System
# ============================================================================

def example_meeting_intelligence():
    """Analyze meeting recordings and transcripts"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Meeting Intelligence System")
    print("="*60)
    
    rag = MultimodalRAGSystem()
    
    # Simulate meeting minutes as text files
    meetings = {
        'team_standup_2024_02_14.txt': '''
            Team Standup - February 14, 2024
            
            Attendees: Alice, Bob, Charlie
            
            Alice: Completed the API documentation. Ready for review.
            Bob: Working on the frontend redesign. 80% complete.
            Charlie: Fixed the database performance issues. Deployed to staging.
            
            Action Items:
            - Alice: Review Bob's design mockups
            - Bob: Complete frontend by Friday
            - Charlie: Monitor staging environment
        ''',
        'client_meeting_2024_02_15.txt': '''
            Client Meeting - February 15, 2024
            
            Client: XYZ Corporation
            Attendees: Alice, Client Rep (John)
            
            Discussion Points:
            - Reviewed Q1 progress
            - Discussed new feature requirements
            - Budget increase approved for Q2
            
            Decisions:
            - Launch date: March 30, 2024
            - Additional resources allocated
            - Weekly sync meetings scheduled
        '''
    }
    
    # Ingest meetings
    for filename, content in meetings.items():
        with open(filename, 'w') as f:
            f.write(content)
        rag.ingest_document(filename)
        os.remove(filename)
    
    # Query the meetings
    queries = [
        "What action items were assigned?",
        "Who completed the API documentation?",
        "What was decided about the launch date?",
        "What is the status of the frontend redesign?"
    ]
    
    for query in queries:
        print(f"\nQ: {query}")
        results = rag.search(query, top_k=1)
        if results:
            print(f"A: From {results[0]['metadata']['source_file']}:")
            print(f"   {results[0]['content'][:150]}...")


# ============================================================================
# EXAMPLE 4: Document Comparison
# ============================================================================

def example_document_comparison():
    """Compare information across multiple documents"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Document Comparison")
    print("="*60)
    
    rag = MultimodalRAGSystem()
    
    # Create comparison documents
    docs = {
        'product_v1_spec.txt': '''
            Product V1 Specification
            - Storage: 128GB
            - RAM: 8GB
            - Display: 13-inch
            - Battery: 8 hours
            - Price: $999
        ''',
        'product_v2_spec.txt': '''
            Product V2 Specification
            - Storage: 256GB (upgraded)
            - RAM: 16GB (upgraded)
            - Display: 14-inch (upgraded)
            - Battery: 12 hours (improved)
            - Price: $1,299
        '''
    }
    
    for filename, content in docs.items():
        with open(filename, 'w') as f:
            f.write(content)
        rag.ingest_document(filename)
        os.remove(filename)
    
    # Compare features
    queries = [
        "How much RAM does the product have?",
        "What is the battery life?",
        "Compare storage capacity",
        "What is the price difference?"
    ]
    
    for query in queries:
        print(f"\n{query}")
        results = rag.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata']['source_file']}: {result['content'][:80]}...")


# ============================================================================
# EXAMPLE 5: Knowledge Base Q&A
# ============================================================================

def example_knowledge_base():
    """Build a question-answering system from documentation"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Knowledge Base Q&A")
    print("="*60)
    
    rag = MultimodalRAGSystem()
    
    # Create knowledge base articles
    kb_articles = {
        'kb_authentication.txt': '''
            Authentication Guide
            
            Q: How do I log in?
            A: Use your email and password at /login
            
            Q: What if I forget my password?
            A: Click "Forgot Password" and check your email
            
            Q: How do I enable 2FA?
            A: Go to Settings > Security > Enable Two-Factor Authentication
        ''',
        'kb_api.txt': '''
            API Documentation
            
            Endpoint: POST /api/search
            Description: Search across documents
            
            Parameters:
            - query (string): Search query
            - top_k (int): Number of results
            
            Response:
            - results (array): Matching documents
            - relevance (float): Match score
        ''',
        'kb_troubleshooting.txt': '''
            Troubleshooting Guide
            
            Issue: Application won't start
            Solution: Check if all dependencies are installed
            
            Issue: Slow search performance
            Solution: Enable GPU acceleration or reduce index size
            
            Issue: Upload fails
            Solution: Verify file size is under 50MB limit
        '''
    }
    
    for filename, content in kb_articles.items():
        with open(filename, 'w') as f:
            f.write(content)
        rag.ingest_document(filename)
        os.remove(filename)
    
    # Interactive Q&A
    print("\nKnowledge Base loaded! Try asking questions:")
    
    sample_questions = [
        "How do I reset my password?",
        "What is the API endpoint for search?",
        "Why is my application slow?"
    ]
    
    for question in sample_questions:
        print(f"\nUser: {question}")
        results = rag.search(question, top_k=1)
        
        if results:
            print(f"Bot: {results[0]['content'][:200]}...")
            print(f"     (Source: {results[0]['metadata']['source_file']})")


# ============================================================================
# EXAMPLE 6: Content Discovery & Recommendation
# ============================================================================

def example_content_discovery():
    """Discover related content based on query"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Content Discovery")
    print("="*60)
    
    rag = MultimodalRAGSystem()
    
    # Create content library
    library = {
        'article_ml_basics.txt': 'Machine learning fundamentals and neural networks',
        'article_nlp.txt': 'Natural language processing and transformers',
        'article_cv.txt': 'Computer vision and convolutional neural networks',
        'article_rl.txt': 'Reinforcement learning and decision making',
        'article_dl.txt': 'Deep learning architectures and training techniques'
    }
    
    for filename, content in library.items():
        with open(filename, 'w') as f:
            f.write(content)
        rag.ingest_document(filename)
        os.remove(filename)
    
    # Discover related content
    interests = [
        "I want to learn about neural networks",
        "What content covers transformers?",
        "Show me articles about training models"
    ]
    
    for interest in interests:
        print(f"\nInterest: {interest}")
        results = rag.search(interest, top_k=3)
        
        print("Recommended content:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata']['source_file']} "
                  f"({result['relevance']:.0%} match)")


# ============================================================================
# EXAMPLE 7: Statistical Analysis
# ============================================================================

def example_statistics():
    """Analyze system statistics and performance"""
    print("\n" + "="*60)
    print("EXAMPLE 7: System Statistics")
    print("="*60)
    
    rag = MultimodalRAGSystem()
    
    # Get statistics
    stats = rag.get_statistics()
    
    print("\nSystem Statistics:")
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Text Chunks: {stats['text_chunks']}")
    print(f"  Image Chunks: {stats['image_chunks']}")
    print(f"  Audio Chunks: {stats['audio_chunks']}")
    print(f"  Storage Path: {stats['storage_path']}")
    
    # Export index
    print("\nExporting index for backup...")
    rag.export_index("rag_backup.json")
    print("✓ Index exported to rag_backup.json")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("MULTIMODAL RAG SYSTEM - EXAMPLE DEMONSTRATIONS")
    print("="*60)
    
    examples = [
        ("Basic Search", example_basic_search),
        ("Cross-Modal Search", example_cross_modal_search),
        ("Meeting Intelligence", example_meeting_intelligence),
        ("Document Comparison", example_document_comparison),
        ("Knowledge Base Q&A", example_knowledge_base),
        ("Content Discovery", example_content_discovery),
        ("System Statistics", example_statistics),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n⚠️  Example '{name}' encountered an error: {e}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
