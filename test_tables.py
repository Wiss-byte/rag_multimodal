import pdfplumber
import pymupdf

PDF = "data/attention.pdf"
doc = pymupdf.open(PDF)


def n_mupdf(page, strategy):
    try:
        return len(page.find_tables(strategy=strategy).tables)
    except Exception:
        return "err"


with pdfplumber.open(PDF) as plumb:
    for i, page in enumerate(plumb.pages, 1):
        has_table_word = "Table" in (page.extract_text() or "")
        pl_lt = len(page.find_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines"}))
        pl_tt = len(page.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text"}))
        pl_xt = len(page.find_tables({"vertical_strategy": "text", "horizontal_strategy": "lines"}))
        mu_l = n_mupdf(doc[i - 1], "lines")
        mu_t = n_mupdf(doc[i - 1], "text")
        print(f"p{i:>2} | mot 'Table': {str(has_table_word):5} | pdfplumber lines/lines={pl_lt} text/text={pl_tt} text/lines={pl_xt} | pymupdf lines={mu_l} text={mu_t}")