class Character(object):
  def __init__(self):
    self.life = 1000
  def attacked(self):
    self.life -= 10
  def __str__(self):
    output = str(self.life) + "만큼 피가 남았음."
    return output


#main.py
import ExampleCode.oop as oop

test = oop.Character()
test.attacked()

print(test, "도망가잇")