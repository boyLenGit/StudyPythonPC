#HTML下的format(dic1)
template = '''<html>
... <head><title>{title}</title></head>
... <body>
... <h1>{title}</h1>
... <p>{text}</p>
... </body>'''
data = {'title': 'My Home Page', 'text': 'Welcome to my home page!'}
print(template.format_map(data))