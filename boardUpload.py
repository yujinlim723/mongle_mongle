from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Dream(db.Model):
    __tablename__ = 'dreams'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)  # 로컬 시간 저장
    
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

def add_dream_to_db(title, content):
    if not title or not content:
        return {'error': '제목과 내용은 필수입니다'}, 400

    try:
        dream = Dream(title=title, content=content)
        db.session.add(dream)
        db.session.commit()
        return {'message': '꿈이 성공적으로 저장되었습니다'}
    except Exception as e:
        db.session.rollback()
        return {'error': f'데이터 저장 중 오류 발생: {e}'}, 500
