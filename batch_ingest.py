"""
Batch Ingestion Utility for Multimodal RAG System
Automatically discover and index all documents in a directory
"""

import os
import sys
from pathlib import Path
from typing import Set, List
import argparse
from tqdm import tqdm
import logging

from multimodal_rag_system import MultimodalRAGSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchIngester:
    """Utility class for batch document ingestion"""
    
    SUPPORTED_EXTENSIONS = {
        'text': {'.pdf', '.docx', '.doc', '.txt'},
        'image': {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'},
        'audio': {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
    }
    
    def __init__(self, rag_system: MultimodalRAGSystem):
        self.rag = rag_system
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'chunks_created': 0
        }
        self.failed_files = []
    
    def is_supported(self, filepath: Path) -> bool:
        """Check if file type is supported"""
        ext = filepath.suffix.lower()
        for extensions in self.SUPPORTED_EXTENSIONS.values():
            if ext in extensions:
                return True
        return False
    
    def get_file_type(self, filepath: Path) -> str:
        """Determine file type category"""
        ext = filepath.suffix.lower()
        for file_type, extensions in self.SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return file_type
        return 'unknown'
    
    def discover_files(self, directory: str, recursive: bool = True) -> List[Path]:
        """Discover all supported files in directory"""
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = []
        
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'
        
        for filepath in directory.glob(pattern):
            if filepath.is_file() and self.is_supported(filepath):
                files.append(filepath)
        
        return sorted(files)
    
    def ingest_file(self, filepath: Path) -> bool:
        """Ingest a single file"""
        try:
            chunks = self.rag.ingest_document(str(filepath))
            self.stats['successful'] += 1
            self.stats['chunks_created'] += len(chunks)
            logger.info(f"✓ {filepath.name} - {len(chunks)} chunks")
            return True
        
        except Exception as e:
            self.stats['failed'] += 1
            self.failed_files.append((filepath, str(e)))
            logger.error(f"✗ {filepath.name} - Error: {e}")
            return False
    
    def ingest_directory(self, 
                        directory: str,
                        recursive: bool = True,
                        file_types: Set[str] = None) -> dict:
        """
        Ingest all supported files from a directory
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            file_types: Set of file types to include ('text', 'image', 'audio')
                       If None, includes all types
        
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Discovering files in: {directory}")
        files = self.discover_files(directory, recursive)
        
        # Filter by file types if specified
        if file_types:
            files = [f for f in files if self.get_file_type(f) in file_types]
        
        self.stats['total_files'] = len(files)
        logger.info(f"Found {len(files)} supported files")
        
        if not files:
            logger.warning("No supported files found!")
            return self.stats
        
        # Show file type distribution
        type_counts = {}
        for f in files:
            ftype = self.get_file_type(f)
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
        
        logger.info("File distribution:")
        for ftype, count in type_counts.items():
            logger.info(f"  {ftype}: {count} files")
        
        # Process files with progress bar
        logger.info("\nProcessing files...")
        for filepath in tqdm(files, desc="Ingesting documents"):
            self.ingest_file(filepath)
        
        return self.stats
    
    def print_summary(self):
        """Print ingestion summary"""
        print("\n" + "="*60)
        print("INGESTION SUMMARY")
        print("="*60)
        print(f"Total files found:    {self.stats['total_files']}")
        print(f"Successfully indexed: {self.stats['successful']}")
        print(f"Failed:              {self.stats['failed']}")
        print(f"Chunks created:      {self.stats['chunks_created']}")
        print("="*60)
        
        if self.failed_files:
            print("\nFailed Files:")
            for filepath, error in self.failed_files:
                print(f"  - {filepath.name}: {error}")
        
        print()
    
    def export_report(self, output_file: str = "ingestion_report.txt"):
        """Export detailed ingestion report"""
        with open(output_file, 'w') as f:
            f.write("MULTIMODAL RAG SYSTEM - INGESTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total files: {self.stats['total_files']}\n")
            f.write(f"Successful: {self.stats['successful']}\n")
            f.write(f"Failed: {self.stats['failed']}\n")
            f.write(f"Chunks created: {self.stats['chunks_created']}\n\n")
            
            if self.failed_files:
                f.write("FAILED FILES:\n")
                f.write("-" * 60 + "\n")
                for filepath, error in self.failed_files:
                    f.write(f"File: {filepath}\n")
                    f.write(f"Error: {error}\n\n")
        
        logger.info(f"Report exported to: {output_file}")


def main():
    """Command-line interface for batch ingestion"""
    parser = argparse.ArgumentParser(
        description="Batch ingest documents into Multimodal RAG System"
    )
    
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing documents to ingest'
    )
    
    parser.add_argument(
        '--storage-dir',
        type=str,
        default='./rag_data',
        help='RAG system storage directory (default: ./rag_data)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Search subdirectories recursively (default: True)'
    )
    
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['text', 'image', 'audio'],
        help='File types to include (default: all)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    parser.add_argument(
        '--export-report',
        type=str,
        help='Export detailed report to file'
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    logger.info("Initializing Multimodal RAG System...")
    rag = MultimodalRAGSystem(
        storage_dir=args.storage_dir,
        use_gpu=not args.no_gpu
    )
    
    # Create ingester
    ingester = BatchIngester(rag)
    
    # Convert types to set if provided
    file_types = set(args.types) if args.types else None
    
    # Ingest directory
    try:
        stats = ingester.ingest_directory(
            directory=args.directory,
            recursive=args.recursive,
            file_types=file_types
        )
        
        # Print summary
        ingester.print_summary()
        
        # Export report if requested
        if args.export_report:
            ingester.export_report(args.export_report)
        
        # Show system statistics
        system_stats = rag.get_statistics()
        print("\nSYSTEM STATISTICS:")
        print(f"Total indexed documents: {system_stats['total_documents']}")
        print(f"Text chunks: {system_stats['text_chunks']}")
        print(f"Image chunks: {system_stats['image_chunks']}")
        print(f"Audio chunks: {system_stats['audio_chunks']}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
