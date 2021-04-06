import pygame

from config import manual_input_use_joystick, joystick_steering_axis, joystick_throttle_axis


class ManualInput:
  th = 0
  st = 0
  q = 0
  e = 0

  def __init__(self):
    pygame.init()
    pygame.font.init()
    width, height = 128, 128
    self.screen = pygame.display.set_mode((width, height))
    self.joystick = pygame.joystick.Joystick(0)
    self.joystick.init()

  def loop(self, img=None):
    if img is not None:
      rotated_image = pygame.transform.rotate(pygame.surfarray.make_surface(img), -90)
      self.screen.blit(rotated_image, [0, 0])
      pygame.display.flip()
    for event in pygame.event.get():
      # check if the event is the X button
      if event.type == pygame.QUIT:
        # if it is quit the game
        pygame.quit()
        self.q = 1
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_w:
          self.th = 1
        elif event.key == pygame.K_s:
          self.th = -1
        elif event.key == pygame.K_d:
          self.st = 1
        elif event.key == pygame.K_a:
          self.st = -1
        elif event.key == pygame.K_e:
          self.e = (self.e + 1) % 2

      if event.type == pygame.KEYUP:
        if event.key == pygame.K_w:
          self.th = 0
        elif event.key == pygame.K_s:
          self.th = 0
        elif event.key == pygame.K_d:
          self.st = 0
        elif event.key == pygame.K_a:
          self.st = 0

      if manual_input_use_joystick:
        self.st = self.joystick.get_axis(joystick_steering_axis)
        self.th = -self.joystick.get_axis(joystick_throttle_axis)
