from sqlalchemy import TIMESTAMP, Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from database import Base
from table_post import Post
from table_user import User


class Feed(Base):
    __tablename__ = "feed_action"

    user_id = Column(
        Integer, ForeignKey("user.id"), primary_key=True
    )
    user = relationship(User)
    post_id = Column(
        Integer, ForeignKey("post.id"), primary_key=True
    )
    post = relationship(Post)
    action = Column(String)
    time = Column(TIMESTAMP)
