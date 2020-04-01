import os


# getContent is a function that returns a list of contents from a folder specified from a path
# parameters: path => path to the folder
def get_content(path):
    content = os.listdir(path)
    print("Path: " + os.getcwd() + "/" + path)
    print("=> Number of images:", len(content), "is loaded from the directory")
    return content


# Testing - getContent()
get_content('Train/Positive')
get_content('Train/Negative')
get_content('Test/Positive')
get_content('Test/Negative')

