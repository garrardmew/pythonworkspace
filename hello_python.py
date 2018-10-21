class Student():
    name="yaya"
    age=18
    __age1=65

    def __init__(self):
        self.name="junjun"
        self.age=24

    def show(self):
        print(self.age)
        print(self.name)

s = Student()
s.show()

print(Student.name)
print(Student.age)

