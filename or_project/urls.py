"""or_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from downhaul.views import DownhaulView, DownhaulExcelView
from backhaul.views import BackhaulView, BackhaulExcelView
from residence.views import ResidenceView, ResExcelView
from allocate.views import AllocateView, AllocExcelView
from allocate_mec.views import AllocateMecView, AllocMecExcelView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('downhaul', DownhaulView.as_view(), name='downhaul_view'),
    path('backhaul', BackhaulView.as_view(), name='backhaul_view'),
    path('residence', ResidenceView.as_view(), name='residence_view'),
    path('allocate', AllocateView.as_view(), name='allocate_view'),
    path('allocate_mec', AllocateMecView.as_view(), name='allocate_mec_view'),

    path('downhaul/excel', DownhaulExcelView.as_view(), name='downhaul_excel_view'),
    path('backhaul/excel', BackhaulExcelView.as_view(), name='backhaul_excel_view'),
    path('residence/excel', ResExcelView.as_view(), name='residence_excel_view'),
    path('allocate/excel', AllocExcelView.as_view(), name='allocate_excel_view'),
    path('allocate_mec/excel', AllocMecExcelView.as_view(), name='allocate_mec_excel_view'),
]
