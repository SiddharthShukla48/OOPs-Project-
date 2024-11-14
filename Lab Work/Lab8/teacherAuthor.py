class Teacher:
    def description(self):
        print("This is a teacher")

class Author:
    def description(self):
        print("This is an author")


class TutorAuthor(Teacher, Author):
    def show_profession(self):
        Teacher.description(self)
        Author.description(self)

tutor_author = TutorAuthor()
tutor_author.show_profession()
