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
        img_counter = 0
        total_scores = {}
        count_per_category = {}

        for page_index, page in enumerate(pages):
            page_number = page.get("page_number", page_index + 1)

            for img_index, img in enumerate(page["images"]):
                img_counter += 1
                print(f"  ↳ PDF Page {page_number}, Image {img_index + 1} (Global Image {img_counter})")

                prompt = f"""
Now, you will begin to evaluate a student's architecture assignment on the architect {architect_name}.

This is a formal submission for university credit. You are receiving the full document as **images**, so you can directly observe the formatting, embedded images, captions, structure, and layout.

---

### How to Grade:

Your role is to critically assess this university-level submission with academic rigor. These assignments are not informal design exercises — they are formal evaluations that contribute to course credit. The student will receive and revise based on your feedback, so your comments must be clear, constructive, and directly tied to the rubric.

Be Fair and Constructive:
- Acknowledge when a student does something well, but avoid vague praise like “good job” — always explain why it works.
- If something falls short — poor layout, unclear citations, missing sections — call it out. Use phrases like “needs revision” or “this should be improved by…” followed by specific, actionable advice.

Do Not Sugarcoat:
- Don’t assume good intentions compensate for missing elements. Every section is held to the same professional and academic standard.
- If key parts (e.g., APA citations, required image counts, biographical detail) are missing or flawed, say so plainly and reduce the score accordingly.

When Something is Strong, Note It:
- If an image citation is consistent across the document, or if the architectural description is especially well-written, say so.
- Strong layout and professionalism should be highlighted — this helps students understand what to keep and build on.

Prioritize These Elements Above All:
1. Accuracy of Academic Citations
2. Caption and Image Attribution Clarity
3. Clear Distinction Between Interior vs Exterior Images
4. Overall Layout and Visual Professionalism

---

### Additional Clarifications:
- Images are embedded (not just links)
- Captions below images include attribution (URLs or photographer names)
- A student photo and bio appear on Page 2
- Table of Contents is present
- 10 buildings are described
- Redundant links are likely citations, not missing content
- If you see an irrelevant image on the very first few pages, it may be an image of the student themselves. Do not grade that image.

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
Score: n/5

Start your rubric-based evaluation below:
"""
                result = self.call_vision_mixed(prompt, img)

                # Extract scores per category
                lines = result.splitlines()
                prev_line = ""
                for line in lines:
                    if line.startswith("**") and line.endswith("**"):
                        prev_line = line.strip("**").strip()
                    elif line.lower().startswith("score:") and prev_line:
                        try:
                            score = int(line.split("/")[0].replace("Score:", "").strip())
                            total_scores[prev_line] = total_scores.get(prev_line, 0) + score
                            count_per_category[prev_line] = count_per_category.get(prev_line, 0) + 1
                        except:
                            continue

                results.append(
                    f"📄 PDF Page {page_number}, 🖼️ Image {img_index + 1} (Global Image {img_counter}) Evaluation:\n{result}\n"
                )

        # Summarize scores
        summary = "\n--- Final Score Summary ---\n"
        final_score = 0
        max_total = 0
        for cat, score_sum in total_scores.items():
            count = count_per_category[cat]
            avg = score_sum / count
            summary += f"{cat}: {avg:.2f}/5 (averaged over {count} images)\n"
            final_score += avg
            max_total += 5

        summary += f"\nTOTAL SCORE: {final_score:.2f}/{max_total}\n"

        # Add student-centered feedback
        feedback_prompt = """
Please provide a short paragraph that could appear in an examiner's critical evaluation of a student assignment.
- It should be written in a non-objectifying, student-centered language (e.g., using possessive pronouns like "your work," directly addressing the student, and acknowledging their role in the process).
"""
        feedback = self.call_chat([{"role": "user", "content": feedback_prompt}])
        summary += "\n--- Examiner’s Critical Evaluation ---\n" + feedback.strip()

        return "\n".join(results) + "\n" + summary


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

    print("\n--- RUBRIC ---\n")
    print(rubric)

    with open("rubric.txt", "w", encoding="utf-8") as f:
        f.write(rubric)

    pages = grader.extract_text_and_images_per_page(args.submission_pdf)
    combined_evaluation = grader.evaluate_combined_pages(rubric, pages, args.architect_name)

    print("\n✅ --- Final Combined Evaluation ---")
    print(combined_evaluation)

    with open("evaluation_result.txt", "w", encoding="utf-8") as f:
        f.write(combined_evaluation)
    print("✅ Saved evaluation to: evaluation_result.txt")


if __name__ == "__main__":
    main()
