from rest_framework import serializers
from .models import Employee

class EmployeeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Employee
        fields = ['id', 'name', 'role', 'email', 'username', 'satker', 'jabatan', 'is_trained']
        read_only_fields = ['is_trained']