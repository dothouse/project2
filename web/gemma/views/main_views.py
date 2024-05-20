from flask import Blueprint, url_for
from werkzeug.utils import redirect

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def select1():
    return redirect(url_for('summary.open_summary_page'))

#
# @bp.route('/')
# def select1():
#     return redirect(url_for('result.open_result_page'))
