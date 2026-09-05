import os
import io
import base64
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from PIL import Image
import pymupdf as fitz  # Modern PyMuPDF import
import docx
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

class DocumentProcessor:
    """
    Handles extraction, chunking, OCR dispatch, and ordered structural parsing
    for TXT, MD, JSON, CSV, PDF, DOCX, and image files.
    """

    @staticmethod
    def get_image_base64_and_meta(image_bytes: bytes, filename: str = "image.png") -> Dict[str, Any]:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            fmt = img.format or "PNG"
            mime_type = f"image/{fmt.lower()}"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"
        
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64}"
        size_kb = len(image_bytes) / 1024.0

        return {
            "data_uri": data_uri,
            "filename": filename,
            "width": width,
            "height": height,
            "size_kb": size_kb,
            "description": f"<image: {filename}, {width}x{height}, {size_kb:.1f} KB>"
        }

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> List[Dict[str, Any]]:
        if len(text) <= chunk_size:
            return [{"chunk_index": 1, "total_chunks": 1, "text": text}]
        
        chunks = []
        start = 0
        idx = 1
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_content = text[start:end]
            chunks.append({
                "chunk_index": idx,
                "text": chunk_content
            })
            start += (chunk_size - overlap)
            idx += 1

        total = len(chunks)
        for c in chunks:
            c["total_chunks"] = total
        return chunks

    @classmethod
    def process_plain_text(
        cls,
        filepath: str,
        task: str,
        call_model_fn: Callable[[str, Optional[Dict[str, Any]]], str]
    ) -> str:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        chunks = cls.chunk_text(raw_text)
        results = []

        for chunk in chunks:
            prefix = f"[CHUNK {chunk['chunk_index']} of {chunk['total_chunks']}]\n"
            content_to_analyze = f"{prefix}{chunk['text']}"
            res = call_model_fn(f"Task: {task}\n\nDocument Content:\n{content_to_analyze}", None)
            results.append(f"{prefix}{res}")

        return "\n\n".join(results)

    @classmethod
    def process_image(
        cls,
        filepath: str,
        task: str,
        call_model_fn: Callable[[str, Optional[Dict[str, Any]]], str]
    ) -> str:
        with open(filepath, "rb") as f:
            image_bytes = f.read()

        meta = cls.get_image_base64_and_meta(image_bytes, filename=os.path.basename(filepath))
        return call_model_fn(task, meta)

    @classmethod
    def process_pdf(
        cls,
        filepath: str,
        task: str,
        call_model_fn: Callable[[str, Optional[Dict[str, Any]]], str]
    ) -> str:
        doc = fitz.open(filepath)
        page_results = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            display_num = page_num + 1
            native_text = page.get_text().strip()
            image_list = page.get_images(full=True)

            # Check if this page is primarily a scan (little or no native text)
            if len(native_text) < 40:
                pix = page.get_pix(dpi=200)
                img_bytes = pix.tobytes("png")
                meta = cls.get_image_base64_and_meta(img_bytes, filename=f"page_{display_num}_scan.png")
                ocr_prompt = f"Perform complete OCR and extract information based on this task: {task}"
                ocr_res = call_model_fn(ocr_prompt, meta)
                page_results.append(f"[PAGE {display_num} — FULL PAGE OCR]\n{ocr_res}")
            else:
                page_parts = []
                page_parts.append(f"[NATIVE TEXT]\n{native_text}")

                if image_list:
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_img = doc.extract_image(xref)
                        if base_img and "image" in base_img:
                            img_bytes = base_img["image"]
                            if len(img_bytes) > 1024:
                                meta = cls.get_image_base64_and_meta(img_bytes, filename=f"page_{display_num}_img_{img_idx+1}.{base_img.get('ext', 'png')}")
                                if meta["width"] >= 40 and meta["height"] >= 40:
                                    img_task = f"Analyze this image from Page {display_num} in the context of the task: {task}"
                                    img_res = call_model_fn(img_task, meta)
                                    page_parts.append(f"[EMBEDDED IMAGE {img_idx+1}]\n{img_res}")

                combined_page_text = "\n\n".join(page_parts)
                page_prompt = f"Task: {task}\n\nPage Content:\n{combined_page_text}"
                res = call_model_fn(page_prompt, None)
                page_results.append(f"[PAGE {display_num}]\n{res}")

        doc.close()
        return "\n\n".join(page_results)

    @classmethod
    def process_docx(
        cls,
        filepath: str,
        task: str,
        call_model_fn: Callable[[str, Optional[Dict[str, Any]]], str]
    ) -> str:
        doc = docx.Document(filepath)
        blocks = []
        block_idx = 1

        # Walk body elements in XML sequence order
        for child in doc.element.body:
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if text:
                    blocks.append({
                        "type": "text",
                        "content": text
                    })
                
                # Check for inline embedded drawings/images inside paragraph runs
                for r in p.runs:
                    element = r._r
                    blips = element.xpath('.//a:blip/@r:embed')
                    for rId in blips:
                        try:
                            rel = doc.part.rels[rId]
                            if "image" in rel.target_ref:
                                img_part = rel.target_part
                                img_bytes = img_part.blob
                                if len(img_bytes) > 512:
                                    meta = cls.get_image_base64_and_meta(img_bytes, filename=f"docx_img_{block_idx}.png")
                                    blocks.append({
                                        "type": "image",
                                        "meta": meta
                                    })
                                    block_idx += 1
                        except Exception:
                            pass

            elif isinstance(child, CT_Tbl):
                tbl = Table(child, doc)
                table_lines = []
                for row in tbl.rows:
                    row_cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    table_lines.append(" | ".join(row_cells))
                if table_lines:
                    blocks.append({
                        "type": "table",
                        "content": "\n".join(table_lines)
                    })

        # Process blocks in sequence
        formatted_blocks = []
        for i, block in enumerate(blocks, start=1):
            if block["type"] == "text":
                formatted_blocks.append(f"[DOCX BLOCK {i} — TEXT]\n{block['content']}")
            elif block["type"] == "table":
                formatted_blocks.append(f"[DOCX BLOCK {i} — TABLE]\n{block['content']}")
            elif block["type"] == "image":
                meta = block["meta"]
                img_task = f"Analyze and extract all text, data, and details from this embedded document image for the task: {task}"
                analysis = call_model_fn(img_task, meta)
                formatted_blocks.append(f"[DOCX BLOCK {i} — IMAGE ANALYSIS]\n{analysis}")

        combined_doc = "\n\n".join(formatted_blocks)
        overall_prompt = f"Task: {task}\n\nExtracted Document Elements:\n{combined_doc}"
        return call_model_fn(overall_prompt, None)

    @classmethod
    def process(
        cls,
        filepath: str,
        task: str,
        file_type: str,
        call_model_fn: Callable[[str, Optional[Dict[str, Any]]], str]
    ) -> str:
        ext = file_type.lower().lstrip(".")
        if ext in ["txt", "md", "json", "csv", "py", "js", "html", "css"]:
            return cls.process_plain_text(filepath, task, call_model_fn)
        elif ext in ["png", "jpg", "jpeg", "webp", "bmp"]:
            return cls.process_image(filepath, task, call_model_fn)
        elif ext == "pdf":
            return cls.process_pdf(filepath, task, call_model_fn)
        elif ext in ["docx", "doc"]:
            return cls.process_docx(filepath, task, call_model_fn)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
