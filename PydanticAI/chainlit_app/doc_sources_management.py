from fpdf import FPDF
import unicodedata
import os
import chainlit as cl

font_dir = os.path.expanduser("~/.local/share/fonts")
opensans_regular_path = os.path.join(font_dir, "OpenSans-Regular.ttf")
opensans_bold_path = os.path.join(font_dir, "OpenSans-Bold.ttf")
opensans_italic_path = os.path.join(font_dir, "OpenSans-Italic.ttf")


def init_pdf(doc_title):
    normalized_title = unicodedata.normalize("NFKC", doc_title)

    if not os.path.exists(opensans_regular_path):
        print(f"Warning: Font file not found at {opensans_regular_path}")
    if not os.path.exists(opensans_bold_path):
        print(f"Warning: Font file not found at {opensans_bold_path}")
    if not os.path.exists(opensans_italic_path):
        print(f"Warning: Font file not found at {opensans_italic_path}")

    # --- Create PDF object ---
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
   
    if os.path.exists(opensans_regular_path):
        pdf.add_font(family="OpenSans", style="", fname=opensans_regular_path, uni=True)
    else:
        print("OpenSans Regular not added. Using default font for regular style if OpenSans is called.")

    # Add Bold
    if os.path.exists(opensans_bold_path):
        pdf.add_font(family="OpenSans", style="B", fname=opensans_bold_path, uni=True)
    else:
        print("OpenSans Bold not added. Using default font for bold style if OpenSans is called.")

    # Add Italic
    if os.path.exists(opensans_italic_path):
        pdf.add_font(family="OpenSans", style="I", fname=opensans_italic_path, uni=True)
    else:
        print("OpenSans Italic not added. Using default font for italic style if OpenSans is called.")

     # Use DejaVu font for Unicode support
    pdf.set_font("OpenSans", "B", 24)  # Bold font for title
    pdf.multi_cell(0, 10, normalized_title)  # Add title
    pdf.ln(10)  # Add space after title

    return pdf

def write_pdf(pdf, text, is_title = False):
    if is_title:
        normalized_section_title = unicodedata.normalize("NFKC",text)
        pdf.add_page()
        pdf.set_font("OpenSans", "B", 18)  # Bold font for title
        pdf.multi_cell(0, 10, normalized_section_title)  # Add title
        pdf.ln(10)  # Add space after title
    else:
        normalized_section_text = unicodedata.normalize("NFKC",text)
        pdf.set_font("OpenSans", size=12)  # Regular font for content
        pdf.multi_cell(0, 10, normalized_section_text, 0, 'L', False)

    return pdf 

#=== Receives a List of chainlit elements and appends it
def sources_update(elements):
    """
    Appends a list of elements to the list of sources stored in the user's session.
    
    Parameters
    ----------
    elements : list
        A list of elements to be appended to the list of sources.
    """
    
    answer_sources = cl.user_session.get("answer_sources")
    answer_sources.append(elements)
    cl.user_session.set("answer_sources", answer_sources)

#=== Receives a Chainlit Message and appends it
def update_conversation_history(msg):

    """
    Appends a Chainlit Message to the conversation history and resume history stored in the user's session.
    
    Parameters
    ----------
    msg : Chainlit Message
        The message to be appended to the conversation history.
    """
    conversation_history = cl.user_session.get("conversation_history")
    conversation_history.append(msg)
    cl.user_session.set("conversation_history", conversation_history)

    resume_history = cl.user_session.get("resume_history")
    resume_history.append({msg.author : msg.content})
    cl.user_session.set("resume_history", resume_history)


