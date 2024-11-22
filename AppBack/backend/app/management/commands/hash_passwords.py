from django.core.management.base import BaseCommand
from django.contrib.auth.hashers import make_password
from app.models import Employee

class Command(BaseCommand):
    help = 'Hash all existing employee passwords'

    def handle(self, *args, **options):
        updated = Employee.objects.all().update(
            password=make_password('default_password')
        )
        self.stdout.write(
            self.style.SUCCESS(f'Successfully hashed {updated} passwords')
        )