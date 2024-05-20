from flask import Flask


# app 구동을 위한 부분
# 함수이름(create_app)과 return 부분의 이름은 그대로 사용
# def create_app(): 이 만들고자 하는 app의 실행과정을 설명하는
def create_app():

    # app의 이름
    # cmd 옵션에서 set FLASK_APP=gemma로 설정하여서
    # app 이름은 gemma로 설정됨
    app = Flask(__name__)

    from .views import main_views, summary_views, result_views
    import result_url_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(summary_views.bp)
    app.register_blueprint(result_views.bp)
    app.register_blueprint(result_url_views.bp)

    return app  # 변경 불가!