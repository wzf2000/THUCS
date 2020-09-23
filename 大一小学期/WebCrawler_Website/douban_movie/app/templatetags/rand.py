from django import template
import random

register = template.Library()

@register.filter
def rand_style(l, r):
    return random.randint(l, r)
