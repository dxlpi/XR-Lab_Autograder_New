import os
import io
import base64
import argparse
from typing import List

import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI


def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def compress_and_encode(pixmap, size=(512, 512)):
    img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return encode_image(buf.getvalue())


class Autograder:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def extract_text_and_images_per_page(self, pdf_path):
        print(f"→ Extracting page-wise text and images from: {pdf_path}")
        doc = fitz.open(pdf_path)
        pages = []

        for page in doc:
            page_text = page.get_text()
            images = []
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                image_data = compress_and_encode(pix)
                images.append(image_data)
            pages.append({"text": page_text, "images": images})
        return pages

    def setup_assignment_context(self, course, assignment_number, assignment_text):
        print("→ Setting up assignment context...")
        prompt = (
            f"You will be evaluating student submission as an autograder for {course}. "
            f"Here is instruction for Assignment {assignment_number}:\n{assignment_text}\n"
            "Please answer the following in order:\n"
            "1. What common problem do you envision among student submissions?\n"
            "2. What common problem could be minimized?\n"
            "3. What are the essential advice, FAQs for students to better understand for this assignment?\n"
            "4. Visual reasoning is required for grading images, what are you looking for for good quality images?\n"
            "   * High resolution\n"
            "   * Spatial clarity\n"
            "   * Clear perspective\n"
            "   * Understandable shapes\n"
            "   * Not visually confusing\n"
            "5. How are you going to break each part based on quantitative and qualitative evaluation?"
        )
        return self.call_chat([{"role": "system", "content": prompt}])

    def generate_rubric(self, context_summary):
        print("→ Generating rubric...")
        system_prompt = (
            "Based on the assignment context and the issues identified, produce a detailed rubric "
            "with clear criteria and scoring guidelines out of 5 points for each category."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_summary}
        ]
        return self.call_chat(messages)

    def call_chat(self, messages):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print("⚠️ API call failed:", e)
            return "[ERROR: Chat API failed]"

    def call_vision_mixed(self, prompt_text, image_b64):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print("⚠️ Vision API call failed:", e)
            return "[ERROR: Vision API failed]"

    def evaluate_combined_pages(self, rubric, pages: List[dict], architect_name: str):
        print("→ Evaluating pages with associated images and text...")
        results = []
        for i, page in enumerate(pages):
            for j, img in enumerate(page["images"]):
                print(f"  ↳ Page {i+1}, Image {j+1}")
                prompt = f"""
Now, you will begin to evaluate a student's architecture assignment on the architect {architect_name}.

This is a formal submission for university credit. You are receiving the full document as **images**, so you can directly observe the formatting, embedded images, captions, structure, and layout.
---
###  How to Grade:
Your role is to critically assess this university-level submission with academic rigor. These assignments are not informal design exercises — they are formal evaluations that contribute to course credit. The student will receive and revise based on your feedback, so your comments must be clear, constructive, and directly tied to the rubric.
Be Fair and Constructive
    - Acknowledge when a student does something well, but avoid vague praise like “good job” — always explain why it works.
    - If something falls short — poor layout, unclear citations, missing sections — call it out. Use phrases like “needs revision” or “this should be improved by…” followed by specific, actionable advice.
Do Not Sugarcoat
    - Don’t assume good intentions compensate for missing elements. Every section is held to the same professional and academic standard.
    - If key parts (e.g., APA citations, required image counts, biographical detail) are missing or flawed, say so plainly and reduce the score accordingly.
When Something is Strong, note It
    - If an image citation is consistent across the document, or if the architectural description is especially well-written, say so.
    - Strong layout and professionalism should be highlighted — this helps students understand what to keep and build on.
Prioritize These Elements Above All
    1. Accuracy of Academic Citations
    2. Caption and Image Attribution Clarity
    3. Clear Distinction Between Interior vs Exterior Images
    4. Overall Layout and Visual Professionalism
---
###  Additional Clarifications:
-  Images are embedded (not just links)
-  Captions below images include attribution (URLs or photographer names)
-  A student photo and bio appear on Page 2
-  Table of Contents is present
-  10 buildings are described
-  Redundant links are likely citations, not missing content
-  If you see an unrelevant image on the very first few pages, it may be an image of the student themselves. Do not grade that image.
---
Now consider the following context from the student document:
### Nearby Text:
{page['text'][:1000]}
---
### RUBRIC
{rubric}
Please assess the submission. For every category:
1. Give a **detailed justification** (1–2 paragraphs)
2. Assign a score **out of 5** based on the detailed rubric below
Format:
**[Category Name]**
Justification: ...
Score: x/5
Start your rubric-based evaluation below:
"""
                result = self.call_vision_mixed(prompt, img)
                results.append(f"Page {i+1} Image {j+1} Evaluation:\n{result}\n")
        return "\n".join(results)


def main():
    parser = argparse.ArgumentParser(description="Run the multimodal ChatGPT-based autograder.")
    parser.add_argument("--assignment_pdf", required=True)
    parser.add_argument("--submission_pdf", required=True)
    parser.add_argument("--architect_name", required=True)
    parser.add_argument("--course", default="COGS 160, Cognitive/Neuroscience for Architecture")
    parser.add_argument("--assignment_number", default="X")
    args = parser.parse_args()

    grader = Autograder()

    print("✔️ Autograder initialized")
    assignment_text = "\n".join([p["text"] for p in grader.extract_text_and_images_per_page(args.assignment_pdf)])
    context_summary = grader.setup_assignment_context(args.course, args.assignment_number, assignment_text)
    rubric = grader.generate_rubric(context_summary)

    pages = grader.extract_text_and_images_per_page(args.submission_pdf)
    combined_evaluation = grader.evaluate_combined_pages(rubric, pages, args.architect_name)

    print("\n✅ --- Final Combined Evaluation ---")
    print("\n--- RUBRIC ---\n")
    print(rubric)
    print("\n--- PER-PAGE IMAGE + TEXT EVALUATIONS ---\n")
    print(combined_evaluation)


if __name__ == "__main__":
    main()
