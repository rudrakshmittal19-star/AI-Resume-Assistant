#!/usr/bin/env python3
"""
AI Resume Assistant (mode-based)

Two modes:
 - ATS Analyzer (default): compares an uploaded resume (PDF) to a job description
     provided either as pasted text or uploaded PDF.
 - Resume Generator: extracts skills from a job description (PDF or pasted text)
     and generates a simple ATS-friendly resume template.

This refactor keeps original NLP algorithms and threading behavior but
converts the UI to a mode-based assistant and reorganizes inputs/results
into panels.

Run:
        python3 app.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import PyPDF2
import nltk
import threading
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize

import spacy
import tkinter.font as tkfont

# ---------------------------
# Load NLP model once (global)
# ---------------------------
print("Loading NLP model...")
nlp = spacy.load("en_core_web_sm")
print("Model loaded successfully!")

# Download necessary NLTK data (if missing)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('words', quiet=True)

# ---------------------------
# Simple SKILL database for exact-match extraction (ATS-friendly)
# ---------------------------
SKILL_DATABASE = {
    "python","java","c++","sql","mysql","pandas","numpy",
    "machine learning","deep learning","data analysis",
    "tensorflow","pytorch","aws","docker","kubernetes",
    "react","node","javascript","excel","power bi",
    "tableau","git","linux","api","cloud","nlp"
}

# ---------------------------
# UI Theme / Style (light white/pink theme)
# ---------------------------
# White background with soft pink panels. Text on pink boxes uses a dark color
ROOT_BG = '#ffffff'            # page background (white)
PANEL_BG = '#ffe6f2'           # light pink for input panels
RESULT_BG = '#ffd9ec'          # slightly different pink for results
BTN_PRIMARY = '#ff85c1'        # primary button pink
BTN_ACTIVE = '#ff5da8'         # active/hover pink
COLOR_SUCCESS = '#16a34a'      # green for good
COLOR_WARNING = '#b45309'      # amber for warning
COLOR_DANGER = '#dc2626'       # red for danger
TEXT_DARK = '#111827'          # dark text used on pink/white backgrounds
FONT_FAMILY = 'Segoe UI'



def extract_skills(text):
    txt = (text or '').lower()
    found = set()
    for skill in SKILL_DATABASE:
        if skill in txt:
            found.add(skill)
    return found


def calculate_ats_score(resume_text, jd_text):
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    matched_skills = resume_skills.intersection(jd_skills)
    missing_skills = jd_skills - resume_skills

    skill_score = (len(matched_skills) / max(len(jd_skills), 1)) * 100

    try:
        if not resume_text.strip() or not jd_text.strip():
            semantic_score = 0.0
        else:
            semantic_score = nlp(resume_text).similarity(nlp(jd_text)) * 100
    except Exception:
        semantic_score = 0.0

    final_score = (0.7 * skill_score) + (0.3 * semantic_score)

    return final_score, matched_skills, missing_skills


def generate_recommendations(missing_skills):
    if not missing_skills:
        return ['Great job — the resume already includes the required skills.']

    recs = []
    for skill in sorted(missing_skills):
        recs.append(f'Add experience with {skill.title()} — list projects or training that used {skill}.')
    return recs


# ---------------------------
# Tech term helpers (unchanged algorithms)
# ---------------------------
initial_tech_terms = [
    "software engineering", "backend development", "database engineering",
    "web development", "devops", "Python", "Django", "MySQL", "PostgreSQL",
    "Amazon Web Services", "AWS", "React", "communication"
]


def find_related_terms(terms, top_n=15):
    related_terms = set()
    try:
        if not (hasattr(nlp.vocab, "vectors") and nlp.vocab.vectors.size):
            raise RuntimeError("No vectors available in the model")
        for term in terms:
            main_doc = nlp(term)
            ms = nlp.vocab.vectors.most_similar(
                main_doc.vector.reshape(1, main_doc.vector.shape[0]), n=top_n)
            for word_id in ms[0][0]:
                related_terms.add(nlp.vocab.strings[word_id])
    except Exception:
        related_terms.update(terms)
        return related_terms

    related_terms.update(terms)
    return related_terms


dynamic_tech_terms = find_related_terms(initial_tech_terms)


def is_tech_related(word, threshold=0.5):
    try:
        token = nlp(word)[0]
        return any(token.similarity(nlp(tech_term)) > threshold for tech_term in dynamic_tech_terms)
    except Exception:
        return False


class ResumeAnalyzerApp:
    """Mode-based AI Resume Assistant GUI using Tkinter."""

    def __init__(self, root):
        self.root = root
        root.title("AI Resume Matcher - NLP Project")
        root.geometry('900x650')
        root.configure(bg=ROOT_BG)

        # Fonts
        self.title_font = tkfont.Font(family=FONT_FAMILY, size=18, weight='bold')
        self.normal_font = tkfont.Font(family=FONT_FAMILY, size=11)

        # State
        self.mode_var = tk.StringVar(value='ats')  # 'ats' or 'gen'
        self.resume_path = None
        self.job_description_path = None
        self.resume_keywords = set()
        self.job_description_keywords = set()

        # Layout: top mode selector, then two frames: input and result
        self._build_mode_selector()
        self._build_main_frames()

    # ---------------------------
    # UI building
    # ---------------------------
    def _build_mode_selector(self):
        top = tk.Frame(self.root, bg=ROOT_BG)
        top.pack(fill='x', pady=(10, 6))

        title = tk.Label(top, text='AI Resume Assistant', bg=ROOT_BG, fg=TEXT_DARK, font=self.title_font)
        title.pack(side='left', padx=12)

        mode_frame = tk.Frame(top, bg=ROOT_BG)
        mode_frame.pack(side='right', padx=12)

        tk.Radiobutton(mode_frame, text='ATS Analyzer', variable=self.mode_var, value='ats', bg=ROOT_BG, fg=TEXT_DARK,
                       selectcolor=ROOT_BG, command=self._on_mode_change).pack(side='left', padx=6)
        tk.Radiobutton(mode_frame, text='Resume Generator', variable=self.mode_var, value='gen', bg=ROOT_BG, fg=TEXT_DARK,
                       selectcolor=ROOT_BG, command=self._on_mode_change).pack(side='left', padx=6)

    def _build_main_frames(self):

        main = tk.Frame(self.root, bg=ROOT_BG)
        main.pack(fill='both', expand=True, padx=12, pady=8)

        # Input Panel (left)
        self.input_panel = tk.LabelFrame(main, text='Input Panel', bg=PANEL_BG, fg=TEXT_DARK, font=self.normal_font)
        self.input_panel.pack(side='left', fill='both', expand=True, padx=(0, 6))

        # Result Panel (right)
        self.result_panel = tk.LabelFrame(main, text='Result Panel', bg=RESULT_BG, fg=TEXT_DARK, font=self.normal_font)
        self.result_panel.pack(side='right', fill='both', expand=True, padx=(6, 0))

        # Inside input panel: Job Description input (text and PDF upload)
        jd_frame = tk.Frame(self.input_panel, bg=PANEL_BG)
        jd_frame.pack(fill='both', expand=False, pady=(8, 6), padx=8)

        jd_label = tk.Label(jd_frame, text='Job Description (paste text or upload PDF):', bg=PANEL_BG, fg=TEXT_DARK)
        jd_label.pack(anchor='w')

        # Large text area for job description
        self.jd_text = tk.Text(jd_frame, height=12, wrap='word', bg='white', fg=TEXT_DARK, insertbackground=TEXT_DARK)
        self.jd_text.pack(fill='both', expand=True, pady=(4, 6))

        jd_btn_frame = tk.Frame(jd_frame, bg=PANEL_BG)
        jd_btn_frame.pack(fill='x')
        tk.Button(jd_btn_frame, text='Upload JD (PDF)', bg=BTN_PRIMARY, fg=TEXT_DARK, activebackground=BTN_ACTIVE, activeforeground=TEXT_DARK, bd=0, relief='raised', padx=10, pady=6, font=self.normal_font, command=self.upload_job_description_pdf).pack(side='left')
        tk.Button(jd_btn_frame, text='Clear JD Text', bg=BTN_PRIMARY, fg=TEXT_DARK, activebackground=BTN_ACTIVE, activeforeground=TEXT_DARK, bd=0, relief='raised', padx=10, pady=6, font=self.normal_font, command=lambda: self.jd_text.delete('1.0', tk.END)).pack(side='left', padx=6)

        # Resume upload area
        resume_frame = tk.Frame(self.input_panel, bg=PANEL_BG)
        resume_frame.pack(fill='x', pady=(6, 6), padx=8)

        resume_label = tk.Label(resume_frame, text='Resume (PDF):', bg=PANEL_BG, fg=TEXT_DARK)
        resume_label.pack(anchor='w')

        self.upload_resume_btn = tk.Button(resume_frame, text='Upload Resume (PDF)', bg=BTN_PRIMARY, fg=TEXT_DARK, activebackground=BTN_ACTIVE, activeforeground=TEXT_DARK, bd=0, relief='raised', padx=10, pady=6, font=self.normal_font, command=self.upload_resume_pdf)
        self.upload_resume_btn.pack(anchor='w', pady=(4, 0))

        # Action buttons
        action_frame = tk.Frame(self.input_panel, bg=PANEL_BG)
        action_frame.pack(fill='x', pady=(8, 6), padx=8)

        self.run_button = tk.Button(action_frame, text='Run', width=20, bg=BTN_PRIMARY, fg=TEXT_DARK, activebackground=BTN_ACTIVE, activeforeground=TEXT_DARK, bd=0, relief='raised', padx=10, pady=6, font=self.normal_font, command=self._on_run_clicked)
        self.run_button.pack(side='left')

        tk.Button(action_frame, text='Clear Results', bg=BTN_PRIMARY, fg=TEXT_DARK, activebackground=BTN_ACTIVE, activeforeground=TEXT_DARK, bd=0, relief='raised', padx=10, pady=6, font=self.normal_font, command=self._clear_results).pack(side='left', padx=6)

        # Result panel: output text area
        # Score canvas (top of result panel)
        self.score_canvas = tk.Canvas(self.result_panel, width=220, height=220, bg='white', highlightthickness=0)
        self.score_canvas.pack(pady=(8, 4))

        self.output_text = tk.Text(self.result_panel, height=20, wrap='word', bg='white', fg=TEXT_DARK, insertbackground=TEXT_DARK)
        self.output_text.pack(fill='both', expand=True, padx=8, pady=8)
        self.output_text.configure(state=tk.DISABLED)

        # Initialize mode state
        self._on_mode_change()

    # ---------------------------
    # Mode handling
    # ---------------------------
    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == 'ats':
            # Resume required
            self.upload_resume_btn.config(state=tk.NORMAL)
            self.run_button.config(text='Analyze')
        else:
            # Resume not required for generator
            self.upload_resume_btn.config(state=tk.DISABLED)
            self.run_button.config(text='Generate Resume')

    # ---------------------------
    # File upload handlers
    # ---------------------------
    def upload_job_description_pdf(self):
        path = filedialog.askopenfilename(filetypes=[('PDF files', '*.pdf')])
        if path:
            self.job_description_path = path
            text = self._read_pdf_text(path)
            if text:
                self.jd_text.delete('1.0', tk.END)
                self.jd_text.insert(tk.END, text)
                self.append_output('Job description PDF loaded into text area.\n')

    def upload_resume_pdf(self):
        path = filedialog.askopenfilename(filetypes=[('PDF files', '*.pdf')])
        if path:
            self.resume_path = path
            self.append_output('Resume uploaded: ' + path + '\n')

    def _read_pdf_text(self, path):
        try:
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() if page.extract_text() else ''
            return text
        except Exception as e:
            messagebox.showerror('Error', f'Failed to read PDF: {e}')
            return ''

    # ---------------------------
    # Core NLP helpers (reuse existing algorithms)
    # ---------------------------
    def extract_keywords_from_text(self, text, threshold=0.5):
        return extract_skills(text)

    def extract_keywords_from_pdf(self, path, threshold=0.5):
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() if page.extract_text() else ''
        return extract_skills(text)

    def find_skill_gap(self):
        matched_skills = self.resume_keywords.intersection(self.job_description_keywords)
        missing_skills = set(self.job_description_keywords) - set(self.resume_keywords)
        return matched_skills, missing_skills

    def calculate_score(self):
        if not self.job_description_keywords:
            self.append_output('No job description keywords available.\n')
            return 0.0

        total_similarity = 0.0
        for jd_keyword in self.job_description_keywords:
            jd_doc = nlp(jd_keyword)
            max_similarity = 0.0
            for resume_keyword in self.resume_keywords:
                resume_doc = nlp(resume_keyword)
                similarity = jd_doc.similarity(resume_doc)
                if similarity > max_similarity:
                    max_similarity = similarity
            total_similarity += max_similarity

        if len(self.job_description_keywords) == 0:
            semantic_score = 0.0
        else:
            semantic_score = (total_similarity / len(self.job_description_keywords)) * 100

        matched = self.resume_keywords.intersection(self.job_description_keywords)
        keyword_match_ratio = len(matched) / max(len(self.job_description_keywords), 1)

        final_score = (semantic_score * 0.7) + (keyword_match_ratio * 100 * 0.3)
        return final_score

    # ---------------------------
    # Run / threading
    # ---------------------------
    def _on_run_clicked(self):
        mode = self.mode_var.get()
        if mode == 'ats':
            jd_text = self.jd_text.get('1.0', tk.END).strip()
            # If JD not provided but resume uploaded -> run resume-only ATS scoring
            if not jd_text and not self.job_description_path:
                if self.resume_path:
                    self.append_output('Running resume-only ATS analysis... Please wait\n')
                    thread = threading.Thread(target=self._run_resume_only_analysis, daemon=True)
                    thread.start()
                    return
                else:
                    messagebox.showwarning('Input required', 'Please provide a job description (text or PDF) or upload a resume for resume-only analysis.')
                    return
            if not self.resume_path:
                messagebox.showwarning('Input required', 'Please upload a resume PDF for ATS analysis.')
                return
            # Clear any existing score meter when running a JD-based ATS analysis
            self.root.after(0, lambda: self.score_canvas.delete('all'))
            self.append_output('Analyzing... Please wait\n')
            thread = threading.Thread(target=self._run_ats_analysis, daemon=True)
            thread.start()
        else:
            jd_text = self.jd_text.get('1.0', tk.END).strip()
            if not jd_text and not self.job_description_path:
                messagebox.showwarning('Input required', 'Please provide a job description (text or PDF).')
                return
            # Clear any existing score meter when generating a resume
            self.root.after(0, lambda: self.score_canvas.delete('all'))
            self.append_output('Generating resume... Please wait\n')
            thread = threading.Thread(target=self._run_resume_generation, daemon=True)
            thread.start()

    def _run_ats_analysis(self):
        try:
            jd_text = self.jd_text.get('1.0', tk.END).strip()
            if jd_text:
                jd_full_text = jd_text
            elif self.job_description_path:
                jd_full_text = self._read_pdf_text(self.job_description_path)
            else:
                jd_full_text = ''

            resume_full_text = self._read_pdf_text(self.resume_path) if self.resume_path else ''

            self.job_description_keywords = extract_skills(jd_full_text)
            self.resume_keywords = extract_skills(resume_full_text)

            out = []
            out.append('Resume Skills:\n' + (', '.join(sorted(self.resume_keywords)) or 'None') + '\n')
            out.append('Job Description Skills:\n' + (', '.join(sorted(self.job_description_keywords)) or 'None') + '\n')
            self.root.after(0, lambda: self._display_output('\n'.join(out)))

            score, matched, missing = calculate_ats_score(resume_full_text, jd_full_text)

            if score >= 85:
                category = 'Strong ATS Match'
            elif score >= 65:
                category = 'Good Match'
            elif score >= 45:
                category = 'Needs Improvement'
            else:
                category = 'Low ATS Compatibility'

            self.root.after(0, lambda: self._display_score_and_category(score, category))

            def show_gaps_and_feedback():
                if matched:
                    self.append_output('Matched Skills:\n' + ', '.join(sorted(matched)) + '\n')
                else:
                    self.append_output('Matched Skills:\nNone\n')

                if missing:
                    self.append_output('Missing Skills:\n' + ', '.join(sorted(missing)) + '\n')
                else:
                    self.append_output('Missing Skills:\nNone\n')

                recs = generate_recommendations(missing)
                self.append_output('\nRecommendations:\n')
                for r in recs:
                    self.append_output('- ' + r + '\n')

            self.root.after(0, show_gaps_and_feedback)

        except Exception as e:
            self.root.after(0, lambda: self.append_output(f'Error during ATS analysis: {e}\n'))

    def _run_resume_generation(self):
        try:
            jd_text = self.jd_text.get('1.0', tk.END).strip()
            if jd_text:
                jd_full_text = jd_text
            elif self.job_description_path:
                jd_full_text = self._read_pdf_text(self.job_description_path)
            else:
                jd_full_text = ''

            self.job_description_keywords = extract_skills(jd_full_text)

            template_lines = []
            template_lines.append('---- Generated ATS-friendly Resume ----\n')
            template_lines.append('Name: [Your Name]\n')
            template_lines.append('Email: [your.email@example.com] | Phone: [XXX-XXX-XXXX]\n')
            template_lines.append('\nProfessional Summary:\n')
            skills_sample = ', '.join(sorted(list(self.job_description_keywords))[:6]) if self.job_description_keywords else 'relevant technologies'
            template_lines.append(f'Motivated professional with experience in {skills_sample}. Demonstrated ability to deliver projects using {skills_sample}.\n')
            template_lines.append('\nSkills:\n')
            if self.job_description_keywords:
                for kw in sorted(self.job_description_keywords):
                    template_lines.append('- ' + kw + '\n')
            else:
                template_lines.append('- [Add relevant skills]\n')

            template_lines.append('\nProjects:\n')
            if self.job_description_keywords:
                sk_list = ', '.join(sorted(list(self.job_description_keywords)))
                template_lines.append(f'- Built a project leveraging {sk_list} to solve business problems and improve efficiency.\n')
                template_lines.append(f'- Implemented pipelines using {skills_sample} and automation to reduce processing time.\n')
            else:
                template_lines.append('- [Project highlighting relevant skills]\n')

            template_lines.append('\nEducation:\n- [Your education details]\n')

            self.root.after(0, lambda: self._display_output('\n'.join(template_lines)))

        except Exception as e:
            self.root.after(0, lambda: self.append_output(f'Error during resume generation: {e}\n'))

    # ---------------------------
    # UI helpers
    # ---------------------------
    def append_output(self, text):
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    # ---------------------------
    # Resume-only ATS scoring helpers
    # ---------------------------
    def _clean_text_spacy(self, text):
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        return ' '.join(tokens)

    def _detect_sections(self, text):
        sections = ['education', 'experience', 'skills', 'projects', 'certifications']
        found = {s: (s in text.lower()) for s in sections}
        return found

    def _detect_experience_years(self, text):
        import re
        matches = re.findall(r"(\d+)\+?\s*(?:years|yrs)", text.lower())
        years = [int(m) for m in matches] if matches else []
        return max(years) if years else 0

    def compute_resume_ats_score(self, resume_text):
        # Skill score: fraction of skills detected from SKILL_DATABASE
        resume_skills = extract_skills(resume_text)
        skill_score = (len(resume_skills) / max(len(SKILL_DATABASE), 1)) * 100

        # Structural score: presence of important sections
        sections = self._detect_sections(resume_text)
        structural_score = (sum(1 for v in sections.values() if v) / len(sections)) * 100

        # Semantic richness: lexical variety using lemmatized unique tokens
        cleaned = self._clean_text_spacy(resume_text)
        tokens = cleaned.split()
        semantic_richness = (len(set(tokens)) / max(len(tokens), 1)) * 100

        # Experience bonus: scale years found into 0-100
        years = self._detect_experience_years(resume_text)
        experience_bonus = min((years / 10) * 100, 100)

        final_score = (0.4 * skill_score) + (0.3 * structural_score) + (0.2 * semantic_richness) + (0.1 * experience_bonus)
        final_score = max(0.0, min(final_score, 100.0))

        breakdown = {
            'skill_score': skill_score,
            'structural_score': structural_score,
            'semantic_richness': semantic_richness,
            'experience_bonus': experience_bonus,
            'final_score': final_score,
            'detected_skills': resume_skills,
            'sections': sections,
            'years': years
        }
        return breakdown

    def _draw_score_meter(self, score):
        # Clear previous
        self.score_canvas.delete('all')
        w = int(self.score_canvas['width'])
        h = int(self.score_canvas['height'])
        cx = w // 2
        cy = h // 2
        r = min(w, h) // 2 - 10

        # Background circle (subtle pink outline for light theme)
        self.score_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline='#f3b6d9', width=18)

        # Foreground arc
        extent = (score / 100.0) * 360
        if score > 70:
            color = COLOR_SUCCESS
        elif score >= 40:
            color = COLOR_WARNING
        else:
            color = COLOR_DANGER

        # draw arc as wide arc (pieslice with thick outline appearance)
        self.score_canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=90, extent=-extent, style='arc', outline=color, width=18)

        # Percentage text (dark on light/pink background)
        pct_font = tkfont.Font(family=FONT_FAMILY, size=24, weight='bold')
        self.score_canvas.create_text(cx, cy, text=f"{score:.0f}%", fill=TEXT_DARK, font=pct_font)

    def _run_resume_only_analysis(self):
        try:
            resume_full_text = self._read_pdf_text(self.resume_path) if self.resume_path else ''
            if not resume_full_text.strip():
                self.root.after(0, lambda: self.append_output('No resume text found for analysis.\n'))
                return

            breakdown = self.compute_resume_ats_score(resume_full_text)
            score = breakdown['final_score']

            # Update UI: draw meter and show breakdown
            self.root.after(0, lambda: self._draw_score_meter(score))

            out_lines = []
            out_lines.append('Resume ATS Score Summary:\n')
            out_lines.append(f"Overall Score: {score:.2f}%\n")
            out_lines.append(f"Skill Score: {breakdown['skill_score']:.2f}%\n")
            out_lines.append(f"Structural Score: {breakdown['structural_score']:.2f}%\n")
            out_lines.append(f"Semantic Richness: {breakdown['semantic_richness']:.2f}%\n")
            out_lines.append(f"Experience Years Detected: {breakdown['years']} (bonus scaled to {breakdown['experience_bonus']:.2f})\n")
            out_lines.append('\nDetected Skills:\n' + ', '.join(sorted(breakdown['detected_skills'])) + '\n')
            out_lines.append('\nDetected Sections:\n')
            for s, present in breakdown['sections'].items():
                out_lines.append(f"- {s.title()}: {'Yes' if present else 'No'}\n")

            self.root.after(0, lambda: self._display_output('\n'.join(out_lines)))

        except Exception as e:
            self.root.after(0, lambda: self.append_output(f'Error during resume-only analysis: {e}\n'))

    def _display_output(self, text):
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, text + '\n')
        self.output_text.configure(state=tk.DISABLED)

    def _display_score_and_category(self, score, category):
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, f'\nMatch Score: {score:.2f}%\n')
        self.output_text.insert(tk.END, f'Result Category: {category}\n')
        self.output_text.configure(state=tk.DISABLED)

    def _clear_results(self):
        # Clear text output
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete('1.0', tk.END)
        self.output_text.configure(state=tk.DISABLED)
        # Also clear the score/accuracy meter canvas
        try:
            self.score_canvas.delete('all')
        except Exception:
            pass


if __name__ == '__main__':
    root = tk.Tk()
    app = ResumeAnalyzerApp(root)
    root.mainloop()
