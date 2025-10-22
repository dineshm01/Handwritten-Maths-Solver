import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
from sympy import sympify, solve, symbols, Eq, simplify, expand, factor, latex, sqrt
from streamlit_drawable_canvas import st_canvas

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Advanced Handwritten Equation Solver", layout="centered")
st.title("🖊️ Advanced Handwritten Equation Solver")

st.write("""
Upload or draw a handwritten equation.  
Supports **multi-variable equations**, **fractions**, **exponents**, and **common math symbols**.  
You can preview OCR results and **edit text before solving**.
""")

# ----------------- TESSERACT PATH -----------------
# 👇 Update this if your Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------- UPLOAD & CANVAS -----------------
uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=200,
    width=500,
    drawing_mode="freedraw",
    key="canvas"
)

# ----------------- IMAGE PROCESSING -----------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    denoised = cv2.medianBlur(thresh, 3)
    processed = cv2.bitwise_not(denoised)
    return processed

def preview_ocr(img):
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    img_copy = img.copy()
    for i in range(len(d['level'])):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        if int(d['conf'][i]) > 30:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy

# ----------------- EQUATION HANDLING -----------------
def clean_equation_text(text):
    text = text.strip().replace(" ", "").replace("\n","")
    text = text.replace("÷","/").replace("×","*").replace("−","-")
    text = text.replace("²","**2").replace("³","**3")
    text = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", text)
    return text

# Add mode selector in sidebar
mode = st.radio(
    "Choose Solution Mode:",
    ["Quick Answer Only", "Step-by-Step with Explanations"],
    index=1
)

def solve_equation(equation_text):
    if not equation_text:
        st.warning("No text recognized.")
        return

    try:
        equation_text_clean = clean_equation_text(equation_text)

        # -----------------------------
        # Split multiple equations
        # -----------------------------
        raw_equations = re.split(r'[\n,;]+', equation_text_clean)
        equations = [eq.strip() for eq in raw_equations if eq.strip()]

        solved_vars = {}  # store numeric values
        eq_objs = []      # list of Eq objects
        all_vars = set()  # all variables in the system

        # Parse equations into SymPy Eq objects
        for eq in equations:
            if "=" in eq:
                lhs_text, rhs_text = eq.split("=", 1)
                lhs, rhs = sympify(lhs_text), sympify(rhs_text)
            else:
                lhs, rhs = sympify(eq), 0

            eq_objs.append(Eq(lhs, rhs))
            all_vars.update(lhs.free_symbols.union(rhs.free_symbols))

        variables = sorted(list(all_vars), key=lambda s: s.name)
        st.success(f"✅ Equations Parsed: {equations}")
        st.write("### Variables Detected:", [v.name for v in variables])

        # -----------------------------
        # Handle arithmetic expressions
        # -----------------------------
        if len(eq_objs) == 1 and len(variables) == 0:
            expr = eq_objs[0].lhs
            st.success(f"✅ Expression Parsed: {equation_text_clean}")

            if mode == "Quick Answer Only":
                st.write("### Result:")
                st.latex(f"{latex(expr)} = {latex(expr.evalf())}")
            else:
                st.write("### Step-by-step Solution:")
                st.write("➡️ Start with the given expression:")
                st.latex(f"{latex(expr)}")

                simplified = simplify(expr)
                if simplified != expr:
                    st.write("➡️ Simplify the expression:")
                    st.latex(f"{latex(simplified)}")

                st.write("➡️ Final result:")
                st.latex(f"{latex(simplified.evalf())}")
            return

        # -----------------------------
        # Solve multi-variable system at once
        # -----------------------------
        solutions = solve(eq_objs, variables, dict=True)

        if not solutions:
            st.warning("No solution found.")
            return

        # -----------------------------
        # Step-by-step display
        # -----------------------------
        if mode != "Quick Answer Only":
            st.write("### Step-by-step Transformation:")
            for i, eq in enumerate(eq_objs):
                st.write(f"➡️ Equation {i+1}:")
                st.latex(f"{latex(eq.lhs)} = {latex(eq.rhs)}")

            # Single-variable linear/quadratic explanations
            for var in variables:
                # Check if variable exists in a single-variable equation
                expr = sum([eq.lhs - eq.rhs for eq in eq_objs if var in eq.free_symbols], 0)
                if expr != 0:
                    deg = expr.as_poly(var).degree() if expr.as_poly() else None
                    if deg == 1:
                        coeff = expr.coeff(var)
                        const = expr - coeff*var
                        st.write(f"➡️ Solve for {var}:")
                        st.latex(f"{latex(var)} = {latex(-const/coeff)}")
                        solved_vars[var] = -const/coeff
                    elif deg == 2:
                        poly = expr.as_poly(var)
                        a = poly.coeff_monomial(var**2)
                        b = poly.coeff_monomial(var)
                        c = poly.coeff_monomial(1)
                        st.write(f"➡️ Quadratic equation for {var}:")
                        st.latex(f"{latex(a*var**2 + b*var + c)} = 0")
                        sol_list = solve(expr, var)
                        for s in sol_list:
                            st.write(f"➡️ Solve for {var}:")
                            st.latex(f"{latex(var)} = {latex(s)}")
                        solved_vars[var] = sol_list[0]

        # -----------------------------
        # Display final solution
        # -----------------------------
        st.write("### ✅ Final variable values:")
        for sol in solutions:
            for var, val in sol.items():
                # Substitute previously solved variables to get numeric value
                val_numeric = val.subs(solved_vars)
                st.latex(f"{latex(var)} = {latex(val_numeric)}")
                solved_vars[var] = val_numeric

    except Exception as e:
        st.error("Could not solve equations: " + str(e))

# ----------------- PROCESS IMAGE -----------------
def process_image(img):
    processed_img = preprocess_image(img)
    preview_img = preview_ocr(processed_img)
    st.image(preview_img, caption="OCR Preview (green boxes)", use_container_width=True)

    ocr_text = pytesseract.image_to_string(processed_img, config="--psm 6")
    ocr_text = ocr_text.strip().replace("\n","").replace(" ","")

    st.write("### OCR Recognized Text (Editable):")
    equation_text = st.text_input("Edit equation before solving", value=ocr_text)

    if st.button("Solve Equation"):
        solve_equation(equation_text)

# ----------------- RUN APP -----------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    process_image(img)

if canvas_result.image_data is not None:
    img_canvas = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    if img_canvas.sum() < (255 * img_canvas.size):
        process_image(img_canvas)
