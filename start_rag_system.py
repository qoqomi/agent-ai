"""
RAG 시스템 (HuggingFace Embeddings 사용)
- PDF 문서 로드
- 벡터 스토어 구축
- 검색 기능
"""
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 환경 변수 로드
load_dotenv()


class RAGSystem:
    """RAG 시스템 클래스"""
    
    def __init__(self, doc_dir: str = "documents"):
        """
        초기화
        
        Args:
            doc_dir: PDF 문서가 있는 폴더 경로
        """
        self.doc_dir = Path(doc_dir)
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore = None
       
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",  # 한글 최적화 모델
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def load_documents(self) -> List:
        """
        documents/ 폴더의 모든 PDF 로드
        
        Returns:
            문서 리스트
        """
        print("\n[INFO] PDF 문서 로드 중...")
        
        docs = []
        pdf_files = sorted(self.doc_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"[WARN] {self.doc_dir} 폴더에서 PDF 파일을 찾지 못했습니다. 빈 결과를 반환합니다.")
            return []
        
        for pdf_path in pdf_files:
            print(f"  [LOADING] {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            loaded_docs = loader.load()
            
            # 메타데이터에 파일명 추가
            for doc in loaded_docs:
                doc.metadata["source_file"] = pdf_path.name
            
            docs.extend(loaded_docs)
        
        print(f"   [SUCCESS] 총 {len(docs)}개 페이지 로드 완료")
        return docs
    
    def build(self):
        """
        벡터 스토어 구축
        1. PDF 로드
        2. 텍스트 분할
        3. 벡터화
        """
        print("\n[INFO] RAG 시스템 구축 중...\n")
        
        # 1. 문서 로드
        docs = self.load_documents()
        if not docs:
            print("[WARN] 문서가 없어 벡터 스토어를 구성하지 않습니다.")
            self.vectorstore = None
            return
        
        # 2. 텍스트 분할
        print("\n[INFO] 텍스트 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        print(f"   [SUCCESS] {len(splits)}개 청크 생성 완료")
        
        # 3. 벡터 스토어 생성
        print("\n[INFO] 벡터 스토어 생성 중 (HuggingFace Embeddings)...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        print("   [SUCCESS] 벡터 스토어 생성 완료\n")
    
    def search(self, query: str, k: int = 5) -> str:
        """
        검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            
        Returns:
            검색 결과 텍스트
        """
        if not self.vectorstore:
            return "[INFO] Vector store not initialized. Add PDFs to documents/ and rerun build()."
        
        # 유사도 검색
        docs = self.vectorstore.similarity_search(query, k=k)
        
        # 결과 포맷팅
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            content = doc.page_content[:300]  # 300자로 제한
            results.append(f"[결과 {i} - 출처: {source}]\n{content}...")
        
        return "\n\n---\n\n".join(results)