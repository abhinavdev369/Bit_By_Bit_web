from flask import session

def authenticate(username, password, users_collection):  # Pass collection as parameter
    user = users_collection.find_one({'username': username, 'password': password})
    if user:
        session["user"] = username
        session["role"] = user.get("role", "user")
        return True
    return False

def is_authenticated():
    return "user" in session

def get_role():
    return session.get("role", "user")