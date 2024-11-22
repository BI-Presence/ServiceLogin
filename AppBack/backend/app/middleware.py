from django.shortcuts import redirect
from .models import Employee

def supervisor_required(get_response):
    def middleware(request):
        employee_id = request.session.get('employee_id')
        
        if employee_id:
            employee = Employee.objects.get(id=employee_id)
            if employee.role != 'supervisor':
                return redirect('login')  # Arahkan kembali ke login jika bukan Supervisor

        return get_response(request)

    return middleware
