import os
import numpy as np
import cv2
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .forms import EmployeeForm, PhotoUploadForm
from .models import Employee, UploadedPhoto
from .face_recognition_training import train_face_recognition_model
from django.db.models import Exists, OuterRef
from tensorflow.keras.models import load_model
from django.contrib import messages
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.decorators import user_passes_test
from functools import wraps
from django.utils import timezone
from django.db import transaction
from django.core.exceptions import PermissionDenied
import logging
from .forms import EmployeeForm, PhotoUploadForm, LoginForm
from .models import Employee, UploadedPhoto
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from django.core.exceptions import ObjectDoesNotExist
from .serializers import EmployeeSerializer
import jwt
from django.conf import settings

def about(request):
    return render(request, "about.html")

def login(request):
    return render(request, "login.html")

# Custom decorator to check if user is supervisor
logger = logging.getLogger(__name__)

def supervisor_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.session.get('user_id'):
            messages.warning(request, 'Please login to access this page.')
            return redirect('login')
        try:
            employee = Employee.objects.get(id=request.session['user_id'])
            if employee.role != 'supervisor':
                logger.warning(f"Unauthorized access attempt by {employee.username}")
                messages.error(request, 'Access denied. Supervisor privileges required.')
                return redirect('login')
            # Update last activity
            request.session['last_activity'] = timezone.now().isoformat()
        except Employee.DoesNotExist:
            logger.error(f"Session user_id {request.session.get('user_id')} not found in database")
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def login(request):
    if request.session.get('user_id'):
        return redirect('index')
        
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            
            try:
                with transaction.atomic():
                    employee = Employee.objects.select_for_update().get(username=username)
                    
                    if employee.role != 'supervisor':
                        messages.error(request, 'Access denied. Only supervisors can login.')
                        logger.warning(f"Non-supervisor login attempt: {username}")
                        return render(request, 'login.html', {'form': form})
                    
                    if check_password(password, employee.password):
                        # Update last login and save
                        employee.last_login = timezone.now()
                        employee.save()
                        
                        # Set session data
                        request.session['user_id'] = employee.id
                        request.session['username'] = employee.username
                        request.session['last_activity'] = timezone.now().isoformat()
                        request.session['role'] = employee.role
                        
                        logger.info(f"Successful login: {username}")
                        return redirect('index')
                    else:
                        messages.error(request, 'Invalid username or password')
                        logger.warning(f"Failed login attempt for user: {username}")
                        
            except Employee.DoesNotExist:
                messages.error(request, 'Invalid username or password')
                logger.warning(f"Login attempt with non-existent username: {username}")
    else:
        form = LoginForm()
    
    return render(request, 'login.html', {'form': form})


def logout(request):
    username = request.session.get('username')
    request.session.flush()
    logger.info(f"User logged out: {username}")
    messages.success(request, 'You have been successfully logged out.')
    return redirect('login')

@supervisor_required
def index(request):
    employee = Employee.objects.get(id=request.session['user_id'])
    return render(request, "index.html", {'employee': employee})


# Protect views that require supervisor access
@supervisor_required
def add_employee(request):
    if request.method == 'POST':
        # Membuat instance form berdasarkan POST data
        employee_form = EmployeeForm(request.POST)
        photo_form = PhotoUploadForm(request.POST, request.FILES)

        # Debugging untuk memeriksa validasi form
        if employee_form.is_valid() and photo_form.is_valid():
            try:
                with transaction.atomic():
                    employee = employee_form.save(commit=False)
                    # Hash password sebelum menyimpan
                    employee.password = make_password(employee.password)
                    employee.save()

                    # Mengambil foto yang diunggah
                    photos = request.FILES.getlist('photos')
                    for photo in photos:
                        # Membuat folder berdasarkan role dan nama pegawai
                        role_folder = os.path.join(settings.MEDIA_ROOT, employee.role)
                        user_folder = os.path.join(role_folder, employee.name)

                        if not os.path.exists(user_folder):
                            os.makedirs(user_folder)

                        # Menyimpan file foto
                        file_path = os.path.join(user_folder, photo.name)
                        with open(file_path, 'wb+') as destination:
                            for chunk in photo.chunks():
                                destination.write(chunk)

                        # Menyimpan data foto ke database
                        UploadedPhoto.objects.create(
                            employee=employee,
                            photo=f'{employee.role}/{employee.name}/{photo.name}'  # Hapus 'uploads/' dari jalur
                        )

                    # Mengarahkan kembali ke halaman utama setelah berhasil
                    return redirect('/')
            except Exception as e:
                messages.error(request, f"Terjadi kesalahan saat menyimpan data: {str(e)}")
                return render(request, 'add_user.html', {
                    'employee_form': employee_form, 
                    'photo_form': photo_form
                })
        else:
            # Debugging pesan error validasi form
            print(f"Employee Form Errors: {employee_form.errors}")
            print(f"Photo Form Errors: {photo_form.errors}")
            # Jika form tidak valid, kirim kembali ke template dengan error
            messages.error(request, "Data tidak valid. Silakan periksa kembali form.")
            return render(request, 'add_user.html', {
                'employee_form': employee_form,
                'photo_form': photo_form
            })
    else:
        # Render form kosong jika method adalah GET
        employee_form = EmployeeForm()
        photo_form = PhotoUploadForm()
        return render(request, 'add_user.html', {
            'employee_form': employee_form,
            'photo_form': photo_form
        })
@supervisor_required
def train_data(request):
    # Menggunakan Exists untuk memeriksa apakah employee memiliki foto yang diupload
    employees = Employee.objects.all().annotate(
        has_photos=Exists(UploadedPhoto.objects.filter(employee=OuterRef('pk')))
    )
    return render(request, "train_data.html", {'employees': employees})

@supervisor_required
def train_model_view(request):
    if request.method == 'POST':
        try:
            model_path, labels_file, embeddings_file = train_face_recognition_model()
            
            # Update the employees' status to is_trained=True in the database
            Employee.objects.update(is_trained=True)
            
            return JsonResponse({
                'status': 'success',
                'message': 'Model training completed successfully.',
                'model_path': model_path,
                'labels_file': labels_file,
                'embeddings_file': embeddings_file
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

# Load the model from MEDIA_ROOT when the view is accessed
def load_model_view(request):
    try:
        model_path = os.path.join(settings.MEDIA_ROOT, 'mtcnn_facenet_ann_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        model = load_model(model_path)

        # Load the labels from MEDIA_ROOT
        labels_path = os.path.join(settings.MEDIA_ROOT, 'face_labels.txt')
        with open(labels_path, 'r') as file:
            labels = [line.strip() for line in file]

        return JsonResponse({'status': 'success', 'message': 'Model and labels loaded successfully.'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

@require_POST
@supervisor_required
def delete_employee(request, employee_id):
    try:
        employee = Employee.objects.get(id=employee_id)
        logger.info(f'Deleting employee with ID: {employee_id}')
        employee.delete()
        return JsonResponse({'status': 'success', 'message': 'Employee deleted successfully'})
    except Employee.DoesNotExist:
        logger.error(f'Employee with ID {employee_id} not found')
        return JsonResponse({'status': 'error', 'message': 'Employee not found'})
    except Exception as e:
        logger.error(f'Error while deleting employee: {str(e)}')
        return JsonResponse({'status': 'error', 'message': str(e)})



@api_view(['POST'])
def face_login(request):
    """
    Endpoint for face recognition login
    """
    try:
        username = request.data.get('username')
        employee = Employee.objects.get(username=username)
        
        # Create tokens
        refresh = RefreshToken.for_user(employee)
        tokens = {
            'refresh': str(refresh),
            'access': str(refresh.access_token)
        }
        
        # Add custom claims
        tokens['access_token'] = jwt.encode({
            'user_id': employee.id,
            'username': employee.username,
            'role': employee.role,
            'email': employee.email
        }, settings.SECRET_KEY, algorithm='HS256')
        
        # Serialize employee data
        serializer = EmployeeSerializer(employee)
        
        return Response({
            'status': 'success',
            'message': 'Login successful',
            'tokens': tokens,
            'user': serializer.data
        })
        
    except ObjectDoesNotExist:
        return Response({
            'status': 'error',
            'message': 'User not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def verify_token(request):
    """
    Endpoint to verify JWT token and return user data
    """
    try:
        user = request.user
        serializer = EmployeeSerializer(user)
        return Response({
            'status': 'success',
            'user': serializer.data
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def supervisor_dashboard(request):
    # Add authentication check here
    return render(request, 'dashboard_supervisor.html')

def employee_dashboard(request):
    # Add authentication check here
    return render(request, 'dashboard_employee.html')