from flask import Blueprint, render_template, request


bp = Blueprint('summary', __name__, url_prefix='/summary')

@bp.route('/')
def open_summary_page():



    return render_template('summary/summary.html')
