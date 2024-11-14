class Teacher:
    def description(self):
        print("This is a teacher")


class Author:
    def description(self):
        print("This is an author")


class TeacherAuthor(Teacher, Author):
    def show_profession(self):
        super(Teacher, self).description() 
        super(Author, self).description()   

tutor_author = TeacherAuthor()
tutor_author.show_profession()
