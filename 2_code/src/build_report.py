from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Cm, RGBColor


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "3_report"
DATA_DIR = ROOT / "1_data"
RESULTS_DIR = ROOT / "2_code" / "results"
FILTER_RESULTS = RESULTS_DIR / "filter_experiment"
MAIN_RESULTS = RESULTS_DIR
FONT = "TH Sarabun New"
BLUE = "17365D"
LIGHT_BLUE = "D9EAF7"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def setup_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.7)
    section.bottom_margin = Cm(1.6)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(16)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.line_spacing = 1.0
    for name, size in (("Title", 24), ("Heading 1", 20), ("Heading 2", 18), ("Heading 3", 16)):
        style = doc.styles[name]
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLUE)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.paragraph_format.space_before = Pt(6)
        style.paragraph_format.space_after = Pt(4)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("517354 Digital Signal Processing 2569/1 | G1  •  หน้า ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    for run in footer.runs:
        set_run_font(run, 12)
    return doc


def set_run_font(run, size: float = 16, bold: bool | None = None, color: str | None = None) -> None:
    run.font.name = FONT
    run.font.size = Pt(size)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_paragraph(doc: Document, text: str, *, bold: bool = False, align=None, size: float = 16):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    set_run_font(run, size, bold)
    return p


def add_heading(doc: Document, text: str, level: int = 1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)
    return p


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), fill)
    tc_pr.append(shade)


def add_table(doc: Document, headers: list[str], rows: list[list[object]], widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = str(header)
        shade_cell(cell, BLUE)
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                set_run_font(run, 14, True, "FFFFFF")
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = str(value)
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cells[index].paragraphs:
                paragraph.paragraph_format.space_after = Pt(1)
                for run in paragraph.runs:
                    set_run_font(run, 13)
    if widths:
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_run_font(run, 14, False)
    return p


def add_bullet(doc: Document, text: str):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run(text)
    set_run_font(run, 15)
    return p


def add_figure(doc: Document, image: Path, caption: str, width: float = 6.0):
    if not image.is_file():
        raise FileNotFoundError(image)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(image), width=Inches(width))
    add_caption(doc, caption)


def make_attack_chart(predictions: list[dict[str, str]], output_path: Path) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        condition = row["condition"] if row["label"] == "spoof" else "bonafide"
        grouped[condition].append(row)

    metrics = []
    for condition, rows in grouped.items():
        correct = sum(row["correct"] == "True" for row in rows)
        metrics.append(
            {
                "condition": condition,
                "n": len(rows),
                "correct": correct,
                "accuracy": correct / len(rows),
                "errors": len(rows) - correct,
            }
        )
    metrics.sort(key=lambda row: (row["condition"] == "bonafide", str(row["condition"])))

    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    labels = [str(row["condition"]) for row in metrics]
    scores = [float(row["accuracy"]) for row in metrics]
    colors = ["#4472C4" if label == "bonafide" else "#70AD47" for label in labels]
    ax.bar(labels, scores, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Class / attack")
    ax.set_title("Threshold pipeline accuracy by class and attack")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=9)
    for index, row in enumerate(metrics):
        ax.text(index, float(row["accuracy"]) + 0.025, f"{row['correct']}/{row['n']}", ha="center", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return metrics


def add_cover(doc: Document, students: list[list[object]]) -> None:
    for _ in range(2):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("รายงานโครงงาน")
    set_run_font(run, 22, True, BLUE)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("การตรวจจับเสียงพูดปลอมด้วย Band-pass Filter\nและ MFCC แบบ Threshold")
    set_run_font(run, 22, True, BLUE)
    add_paragraph(doc, "517354 Digital Signal Processing ภาคการศึกษาที่ 1/2569", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "กลุ่ม G1", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=18)
    doc.add_paragraph()
    add_table(doc, ["ลำดับ", "ชื่อ-นามสกุล", "รหัสนักศึกษา", "หน้าที่"], students, [0.55, 1.65, 1.3, 2.45])
    doc.add_paragraph()
    add_paragraph(doc, "วันที่จัดทำ: 4 พฤศจิกายน 2569", align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()


def build_report() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc = setup_document()

    gt = read_rows(DATA_DIR / "ground_truth" / "labels.csv")
    info_path = DATA_DIR / "ground_truth" / "ASVspoof2019_LA_subset_info.txt"
    info = info_path.read_text(encoding="utf-8")
    threshold_metrics = read_rows(MAIN_RESULTS / "threshold_pipeline_metrics.csv")[0]
    predictions = read_rows(MAIN_RESULTS / "threshold_pipeline_predictions.csv")
    failures = read_rows(MAIN_RESULTS / "threshold_pipeline_failure_cases.csv")
    filter_metrics = read_rows(FILTER_RESULTS / "metrics.csv")
    figure_dir = MAIN_RESULTS / "figures"
    attack_chart = figure_dir / "threshold_accuracy_by_attack.png"
    attack_metrics = make_attack_chart(predictions, attack_chart)

    train_n = sum(row["file"].split("/")[1] == "train" for row in gt)
    eval_n = sum(row["file"].split("/")[1] == "eval" for row in gt)
    students = [
        [1, "xxx", "xxx", "xxx"],
        [2, "xxx", "xxx", "xxx"],
    ]

    add_cover(doc, students)

    add_heading(doc, "1. ปัญหาและคำถามวิจัย")
    add_paragraph(
        doc,
        "โครงงานนี้ทดลองว่าสามารถใช้การประมวลผลสัญญาณเสียงแบบง่ายเพื่อแยกเสียงพูดจริง "
        "(bonafide) ออกจากเสียงพูดปลอมที่สร้างด้วยระบบสังเคราะห์เสียงหรือแปลงเสียง (spoof) ได้เพียงใด "
        "คำถามหลักคือ Band-pass 100–4,000 Hz ร่วมกับ MFCC coefficient ลำดับที่ 8 "
        "และกฎ threshold แบบ if/else ให้ผลอย่างไรบนชุด eval",
    )
    add_table(
        doc,
        ["องค์ประกอบ", "รายละเอียด"],
        [
            ["Input", "ไฟล์เสียงพูด FLAC, mono หรือ stereo, sampling rate 16 kHz"],
            ["Output", "หนึ่ง label ต่อไฟล์: bonafide หรือ spoof"],
            ["วิธีหลัก", "Band-pass → MFCC coefficient 8 → threshold → if/else"],
            ["คำถามเชิงทดลอง", "filter/feature ใดให้ผลดีกว่า baseline ไม่กรองและ feature เดี่ยวอื่น"],
        ],
        [1.35, 4.6],
    )
    add_paragraph(
        doc,
        "ASVspoof 2019 LA มีเสียงจริงและเสียงปลอมพร้อม protocol label จึงใช้ label เป็น ground truth "
        "สำหรับการจำแนกแบบมีเฉลยได้ การทดลองนี้เป็นงาน DSP เชิงสำรวจ ไม่ได้มุ่งสร้างระบบป้องกันการปลอมเสียง "
        "ที่พร้อมใช้งานจริง",
    )
    doc.add_page_break()

    add_heading(doc, "2. ชุดข้อมูลและ EDA")
    add_paragraph(
        doc,
        "ใช้ชุด Logical Access (LA) จาก ASVspoof 2019 ตามไฟล์ README และ CM protocol ที่เก็บเป็นแหล่งอ้างอิง "
        "ชุด subset นี้สุ่มแบบสมดุลแยกตาม split และ label ด้วย seed 517354 โดยไม่เปลี่ยนแปลงไฟล์เสียง",
    )
    add_table(
        doc,
        ["Split", "bonafide", "spoof", "รวม"],
        [["train", 2350, 2350, 4700], ["eval", 775, 775, 1550], ["รวม", 3125, 3125, 6250]],
        [1.2, 1.2, 1.2, 1.2],
    )
    add_caption(doc, "ตารางที่ 1 จำนวนไฟล์เสียงใน subset ที่ใช้ในการทดลอง")
    add_paragraph(
        doc,
        "ไฟล์เสียงทั้งหมด 6,250 ไฟล์ใช้ sampling rate 16 kHz และบันทึกแบบ FLAC; ขนาด subset ประมาณ 362.61 MiB "
        "protocol ต้นฉบับระบุคลาส bonafide/spoof และรหัส attack A07–A19 เฉลยใน ground_truth/labels.csv "
        "เชื่อมกับ protocol ด้วย file ID โดยเก็บ README, license และ CM protocols ไว้ใน "
        "ground_truth/source_reference/ เพื่อสอบย้อนกลับ",
    )
    add_paragraph(
        doc,
        "ชุดข้อมูลใช้ Open Data Commons Attribution License ตาม LICENSE ที่แนบมากับฐานข้อมูล "
        "ซึ่งกำหนดการให้ attribution สำหรับฐานข้อมูล; สิทธิ์ของเสียงแต่ละรายการอาจมีเงื่อนไขอื่นตามที่ license เตือนไว้ "
        "จึงเก็บไฟล์ license และอ้าง ASVspoof 2019 เป็นแหล่งข้อมูล",
    )
    add_paragraph(
        doc,
        "EDA พบว่าข้อมูล train/eval ถูกทำให้สมดุล 50:50 เพื่อให้ทำการทดลองง่าย แต่ไม่แทนสัดส่วนจริงของชุดเต็ม "
        "เสียง attack หลายระบบมีลักษณะต่างกัน จึงใช้ attack ID เป็น condition สำหรับแจกแจงผลเพิ่มเติม "
        "และใช้ sampling rate คงที่ 16 kHz เป็นข้อมูลตั้งต้นสำหรับเลือกขอบเขต filter",
    )
    add_caption(doc, "ตารางที่ 2 จำนวนสุ่มถูกกำหนดไว้ใน ground_truth/ASVspoof2019_LA_subset_info.txt")
    doc.add_page_break()

    add_heading(doc, "3. Pipeline และเหตุผล")
    add_figure(
        doc,
        figure_dir / "bandpass_before_after.png",
        "ภาพที่ 1 ตัวอย่าง waveform ก่อนและหลัง Band-pass ของเสียง eval หนึ่งตัวอย่างต่อคลาส",
        5.9,
    )
    add_table(
        doc,
        ["ขั้น", "การทำงาน / พารามิเตอร์", "เหตุผลและสิ่งที่ส่งต่อ"],
        [
            ["1. โหลด", "FLAC → float32; stereo เฉลี่ยเป็น mono; 16 kHz", "รูปแบบ input สม่ำเสมอ"],
            ["2. Band-pass", "Butterworth ลำดับ 4; 100–4,000 Hz; sosfiltfilt", "ลดพลังงานนอกย่านที่กำหนด; ได้ waveform ที่กรองแล้ว"],
            ["3. MFCC", "13 coefficients; n_fft=512; hop=256; เฉลี่ย MFCC ลำดับ 8 ตามเวลา", "ย่อข้อมูล spectrum เป็น feature scalar ต่อไฟล์"],
            ["4. ตัดสิน", "threshold จาก train; ถ้า MFCC8 < -19.426 ให้ spoof มิฉะนั้น bonafide", "กฎ if/else ส่งออก label"],
        ],
        [0.9, 2.6, 2.6],
    )
    add_caption(doc, "ตารางที่ 3 ขั้นตอนและพารามิเตอร์ของ pipeline หลัก")
    add_paragraph(
        doc,
        "ใช้ threshold ที่เลือกบน train ด้วย balanced accuracy เพื่อไม่ให้ขั้นตัดสินอิงคลาสใดคลาสหนึ่งมากเกินไป "
        "ไม่มี dev ใน subset นี้ จึงไม่มีชุดสำหรับปรับค่าระหว่าง train และ eval; eval ถูกใช้ในรอบสำรวจเปรียบเทียบ "
        "ดังนั้นผลต้องตีความเป็น exploratory",
    )
    add_paragraph(
        doc,
        "ขั้นที่ไม่ใช้ใน pipeline หลัก: high-pass/low-pass แบบอื่นและ feature RMS, ZCR, spectral centroid "
        "ทดลองเพื่อเปรียบเทียบ; ไม่ใช้ classifier ML ใน pipeline หลัก เพราะเป้าหมายเน้น filter กับกฎ if/else "
        "การลอง RBF-SVM เป็นผลเสริมและไม่ใช่ผลที่ใช้สรุป pipeline หลัก",
    )
    add_paragraph(
        doc,
        "อ้างอิงโค้ด: 2_code/G00_pipeline.ipynb หัวข้อ 4–5; การสำรวจ threshold ทางเลือกใน "
        "2_code/src/compare_audio_filters.py",
    )
    doc.add_page_break()

    add_heading(doc, "4. ผลของแต่ละขั้นและวิธีที่ไม่ช่วย")
    add_paragraph(
        doc,
        "ผลก่อน pipeline: baseline ไม่กรองร่วมกับ MFCC coefficient 8 ให้ eval balanced accuracy "
        "0.696 และ macro-F1 0.686. เมื่อใช้ low-pass 4,000 Hz ตัวเลขเป็น 0.772/0.772; "
        "เมื่อใช้ band-pass 100–4,000 Hz เป็น 0.775/0.775. ค่าเพิ่มจาก baseline 0.079 "
        "หรือ 7.9 จุดเปอร์เซ็นต์ balanced accuracy ในการทดลองสำรวจนี้",
    )
    add_figure(
        doc,
        figure_dir / "threshold_mfcc_filter_comparison.png",
        "ภาพที่ 2 ผล eval balanced accuracy ของ MFCC threshold เมื่อใช้ filter ต่างกัน",
        5.6,
    )
    mfcc_results = [row for row in filter_metrics if row["feature"].startswith("mfcc_")]
    add_table(
        doc,
        ["Filter + feature", "threshold", "eval balanced accuracy", "macro-F1"],
        [
            [
                f"{row['filter']} + {row['feature']}",
                f"{float(row['threshold']):.3f}",
                f"{float(row['eval_balanced_accuracy']):.3f}",
                f"{float(row['eval_macro_f1']):.3f}",
            ]
            for row in sorted(mfcc_results, key=lambda row: float(row["eval_balanced_accuracy"]), reverse=True)
        ],
        [2.4, 1.0, 1.45, 1.0],
    )
    add_caption(doc, "ตารางที่ 4 ผลของ MFCC threshold แยกตาม filter")
    add_paragraph(
        doc,
        "RMS, ZCR และ spectral centroid แบบ threshold เดี่ยวให้ผลต่ำกว่าการใช้ MFCC8 ในการเปรียบเทียบนี้ "
        "โดย balanced accuracy อยู่ระหว่าง 0.425 ถึง 0.648 ตาม feature/filter (ดู metrics.csv) "
        "จึงไม่เลือก feature เหล่านี้เป็นตัวตัดสินหลัก; ผลนี้หมายถึง feature เดี่ยวกับ threshold ที่ทดลอง "
        "ไม่ใช่ข้อสรุปว่า feature ดังกล่าวไม่มีประโยชน์ใน pipeline อื่น",
    )
    doc.add_page_break()

    add_heading(doc, "5. ผลบนข้อมูล eval ทั้งชุด")
    n_eval = int(threshold_metrics["n_eval"])
    n_correct = int(threshold_metrics["n_correct"])
    n_errors = int(threshold_metrics["n_errors"])
    add_paragraph(
        doc,
        f"เกณฑ์รายชิ้น: ทำนายตรงกับเฉลยเป็น “สำเร็จ”; ทำนายไม่ตรงเป็น “ไม่สำเร็จ” "
        f"เนื่องจาก label เป็นประเภทสองคลาส จึงไม่มีระดับพลาดเล็กน้อยรายไฟล์ "
        f"ผล eval: ถูก {n_correct}/{n_eval} ไฟล์ ({n_correct/n_eval:.1%}), "
        f"ผิด {n_errors}/{n_eval} ไฟล์ ({n_errors/n_eval:.1%}).",
    )
    add_figure(
        doc,
        figure_dir / "threshold_confusion_matrix.png",
        "ภาพที่ 3 Confusion matrix ของ pipeline หลักบน eval",
        3.7,
    )
    tn, fp, fn, tp = 627, 148, 200, 575
    add_table(
        doc,
        ["Metric", "ค่า"],
        [
            ["Accuracy", f"{float(threshold_metrics['eval_accuracy']):.3f}"],
            ["Balanced accuracy", f"{float(threshold_metrics['eval_balanced_accuracy']):.3f}"],
            ["Macro-F1", f"{float(threshold_metrics['eval_macro_f1']):.3f}"],
            ["Bonafide ถูก", f"{tn}/775 ({tn/775:.1%})"],
            ["Spoof ถูก", f"{tp}/775 ({tp/775:.1%})"],
            ["False positive / false negative", f"{fp} / {fn}"],
        ],
        [3.3, 2.4],
    )
    add_caption(doc, "ตารางที่ 5 ตัวชี้วัดและจำนวนทำนายถูก/ผิดบน eval")
    add_paragraph(
        doc,
        "Train balanced accuracy ที่ใช้เลือก threshold เท่ากับ "
        f"{float(threshold_metrics['train_balanced_accuracy']):.3f}; eval balanced accuracy เท่ากับ "
        f"{float(threshold_metrics['eval_balanced_accuracy']):.3f}. "
        "ทั้งสองค่าไม่ควรสับสนกัน เพราะ threshold ถูกเลือกจาก train ขณะที่ eval ใช้ประเมิน",
    )
    doc.add_page_break()

    add_heading(doc, "5. ผลแยกตามสภาวะและการกระจาย")
    add_paragraph(
        doc,
        "กำหนดเกณฑ์สรุประดับ condition เพื่อช่วยอ่านแนวโน้ม: accuracy ≥ 0.80 = สำเร็จ, "
        "0.50–<0.80 = พลาดเล็กน้อย/ผลปานกลาง, และ <0.50 = ไม่สำเร็จใน condition นั้น "
        "เกณฑ์นี้เป็นเกณฑ์รายงานที่กลุ่มกำหนด ไม่ใช่มาตรฐาน benchmark",
    )
    add_figure(
        doc,
        attack_chart,
        "ภาพที่ 4 Accuracy ของ pipeline threshold แยกตาม bonafide และ spoof attack",
        6.0,
    )
    condition_rows = [
        [row["condition"], row["n"], row["correct"], row["errors"], f"{float(row['accuracy']):.3f}"]
        for row in sorted(attack_metrics, key=lambda row: float(row["accuracy"]))
    ]
    add_table(doc, ["Condition", "จำนวน", "ถูก", "ผิด", "Accuracy"], condition_rows, [1.15, 0.8, 0.8, 0.8, 1.0])
    add_caption(doc, "ตารางที่ 6 ผลของ pipeline หลักแยกตาม attack ID/เสียง bonafide")
    add_paragraph(
        doc,
        "ตารางที่ 6 แสดงว่าผลแตกต่างกันมากตาม condition; attack ที่ accuracy ต่ำควรเป็นจุดเน้นในการวิเคราะห์ "
        "ค่าเฉลี่ยรวมอย่างเดียวจึงปกปิดความแตกต่างภายในชุดข้อมูลได้ การสุ่ม subset มีจำนวนต่อ attack ไม่เท่ากัน "
        "จึงควรอ่านทั้ง accuracy และจำนวนตัวอย่างประกอบ",
    )
    doc.add_page_break()

    add_heading(doc, "6. Failure cases")
    add_paragraph(
        doc,
        "failure cases ต่อไปนี้แสดงไฟล์ที่ขั้นตัดสิน threshold ให้ label ผิด "
        "ระบุว่าพังที่ขั้นสุดท้ายได้จากการเปรียบเทียบ MFCC8 ที่คำนวณกับ threshold กับ label จริง "
        "ข้อมูลที่บันทึกไม่ได้ระบุสาเหตุเชิงเสียง/feature ที่ลึกกว่านี้ จึงไม่กล่าวโทษ filter โดยไม่มีหลักฐาน",
    )
    failure_table = [
        [
            index + 1,
            row["id"],
            row["expected"],
            row["predicted"],
            row["attack_id"],
            f"{float(row['mfcc_8']):.2f}",
            row["failed_stage"],
        ]
        for index, row in enumerate(failures[:3])
    ]
    add_table(
        doc,
        ["#", "File ID", "ควรได้", "ได้จริง", "Attack", "MFCC8", "ขั้นที่พลาด"],
        failure_table,
        [0.35, 1.25, 0.8, 0.8, 0.7, 0.7, 1.25],
    )
    add_caption(doc, "ตารางที่ 7 ตัวอย่าง failure cases จำนวน 3 กรณีจาก eval")
    for row in failures[:3]:
        add_bullet(
            doc,
            f"{row['id']}: เฉลย {row['expected']} แต่ทำนาย {row['predicted']}; "
            f"MFCC8={float(row['mfcc_8']):.2f}, threshold=-19.426. "
            "ตรวจได้จากกฎ if/else และผลรายไฟล์ใน threshold_pipeline_predictions.csv",
        )
    add_paragraph(
        doc,
        "วิธีต่อยอดเพื่อหาสาเหตุ: เปิด waveform/spectrogram ของแต่ละไฟล์เทียบกับไฟล์ที่ทายถูกใน condition เดียวกัน "
        "และบันทึกค่า feature ก่อน/หลัง filter เพิ่มเติม งานนี้ยังไม่ได้ทำการวิเคราะห์เชิงลึกดังกล่าว",
    )
    doc.add_page_break()

    add_heading(doc, "7. สรุปและการแบ่งงาน")
    add_paragraph(
        doc,
        "Pipeline หลัก Band-pass 100–4,000 Hz + MFCC coefficient 8 + threshold `if/else` "
        f"ได้ eval balanced accuracy {float(threshold_metrics['eval_balanced_accuracy']):.3f}, "
        f"macro-F1 {float(threshold_metrics['eval_macro_f1']):.3f}, "
        f"ถูก {n_correct}/{n_eval} และพลาด {n_errors}/{n_eval} ไฟล์",
    )
    add_paragraph(
        doc,
        "ขั้นที่ช่วยมากที่สุดในกลุ่มวิธี threshold ที่ลองคือการใช้ band-pass ร่วมกับ MFCC8: "
        "balanced accuracy 0.775 เทียบกับ 0.696 สำหรับ MFCC8 แบบไม่กรอง (+0.079). "
        "Low-pass + MFCC8 ใกล้เคียงที่ 0.772; RMS/ZCR/centroid แบบ threshold เดี่ยวไม่ช่วยเท่า MFCC8 "
        "ในชุดการทดลองนี้",
    )
    add_paragraph(
        doc,
        "ข้อจำกัด: ใช้ subset 6,250 ไฟล์ที่สุ่มสมดุล; ไม่มี dev; เลือก pipeline จากการเปรียบเทียบ eval "
        "จึงเป็นผลสำรวจและอาจ optimistic; threshold แบบ scalar ใช้ข้อมูลเสียงเพียงมิติเดียว; "
        "ผลไม่แทนสัดส่วน class และ attack ของชุดเต็ม; ยังไม่วิเคราะห์ failure เชิง waveform รายกรณี",
    )
    add_heading(doc, "การแบ่งงาน", level=2)
    add_table(
        doc,
        ["สมาชิก", "รหัส", "หน้าที่"],
        [
            ["xxx", "xxx", "xxx"],
            ["xxx", "xxx", "xxx"],
        ],
        [1.8, 1.6, 3.0],
    )
    add_heading(doc, "เอกสารอ้างอิง", level=2)
    add_bullet(doc, "ASVspoof 2019, Logical Access (LA), README.LA.txt และ CM protocol files ที่เก็บใน 1_data/ground_truth/source_reference/.")
    add_bullet(doc, "Open Data Commons. Open Data Commons Attribution License (ODC-By), LICENSE.txt ที่แนบมากับชุดข้อมูล.")
    add_bullet(doc, "รายละเอียดการสุ่ม subset และจำนวนตัวอย่าง: 1_data/ground_truth/ASVspoof2019_LA_subset_info.txt.")
    add_bullet(doc, "โค้ด pipeline และผล: 2_code/G00_pipeline.ipynb; 2_code/src/compare_audio_filters.py.")

    output_docx = REPORT_DIR / "G01_report.docx"
    doc.save(output_docx)
    return output_docx


if __name__ == "__main__":
    print(build_report())
