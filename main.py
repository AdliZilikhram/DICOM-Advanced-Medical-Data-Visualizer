import sys
import os
import json
import pickle
import numpy as np
import time
import vtk
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QMenuBar, QAction, QToolBar,
                             QLabel, QMessageBox, QDialog, QTextEdit, QFileDialog,
                             QListWidget, QListWidgetItem, QSplitter, QProgressBar, QSlider, 
                             QStackedWidget, QColorDialog, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent, QTimer
from PyQt5.QtGui import QPalette, QColor, QMovie
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util.colors import cornflower

class myVTK:
    def __init__(self):
        self.colors = vtk.vtkNamedColors()
        self.actors = []
        self.mappers = []
        self.sources = []
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.slice_planes = {}
        self.is_initialized = False
        self.is_rendering = False
        
        # Medical imaging specific attributes
        self.imageData = None
        self.reader = None
        self.volumeMapper = None
        self.volumeProperty = None
        self.gradientOpacity = vtk.vtkPiecewiseFunction()
        self.scalarOpacity = vtk.vtkPiecewiseFunction()
        self.color = vtk.vtkColorTransferFunction()
        self.volume = None
        self.interactorStyle = None
        
        # Coordinate system actors
        self.coordinate_axes = None
        self.initial_camera_position = None
        
        # Model management
        self.loaded_datasets = {}
        self.current_dataset = None
        
        # Measurement attributes
        self.measurement_points = []
        self.measurement_actors = []
        self.measurement_labels = []
        self.is_measuring = False
        
        # Measurement line properties
        self.measurement_line_source = None
        self.measurement_line_mapper = None
        self.measurement_line_actor = None
        
        # Opacity transfer functions
        self.opacity_function = vtk.vtkPiecewiseFunction()
        self.color_function = vtk.vtkColorTransferFunction()
        
        # Default window/level values
        self.window_value = 400
        self.level_value = 40
        
        self.main_window = None
        self.is_slice_view_active = False

    def create_simple_coordinate_system(self):
        axes = vtk.vtkAxesActor()
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        axes.GetXAxisShaftProperty().SetColor(1, 0, 0)
        axes.GetYAxisShaftProperty().SetColor(0, 1, 0)
        axes.GetZAxisShaftProperty().SetColor(0, 0, 1)
        axes.SetTotalLength(100, 100, 100)
        axes.SetShaftTypeToCylinder()
        return axes

    def create_object(self):
        try:
            if not self.reader:
                print("No reader available for volume rendering")
                return None
                
            self.reader.Update()
            image_data = self.reader.GetOutput()
            if not image_data:
                print("No image data available for volume rendering")
                return None

            # Initialize volume mapper if not exists
            if self.volumeMapper is None:
                self.volumeMapper = vtk.vtkSmartVolumeMapper()
                
            self.volumeMapper.SetInputConnection(self.reader.GetOutputPort())
            self.volumeMapper.SetBlendModeToComposite()
            
            try:
                self.volumeMapper.SetRequestedRenderModeToRayCastAndTexture()
                print("Using CPU ray casting mode")
            except:
                try:
                    self.volumeMapper.SetRequestedRenderModeToDefault()
                    print("Using default rendering mode")
                except:
                    print("Using fallback rendering")
            
            # Initialize volume property if not exists
            if self.volumeProperty is None:
                self.volumeProperty = vtk.vtkVolumeProperty()
                
            self.volumeProperty.ShadeOn()
            self.volumeProperty.SetInterpolationTypeToLinear()
            
            self.scalarOpacity.RemoveAllPoints()
            self.color.RemoveAllPoints()
            self.opacity_function.RemoveAllPoints()
            self.color_function.RemoveAllPoints()
            
            scalar_range = image_data.GetScalarRange()
            print(f"Scalar range: {scalar_range}")
            
            min_val, max_val = scalar_range
            
            # Default opacity function
            self.opacity_function.AddPoint(min_val, 0.0)
            self.opacity_function.AddPoint(min_val + (max_val - min_val) * 0.1, 0.1)
            self.opacity_function.AddPoint(min_val + (max_val - min_val) * 0.3, 0.8)
            self.opacity_function.AddPoint(max_val, 1.0)
            
            # Default color function
            self.color_function.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
            self.color_function.AddRGBPoint(min_val + (max_val - min_val) * 0.3, 0.5, 0.5, 0.5)
            self.color_function.AddRGBPoint(min_val + (max_val - min_val) * 0.7, 1.0, 1.0, 1.0)
            
            # Use the piecewise functions
            self.scalarOpacity = self.opacity_function
            self.color = self.color_function
            
            self.volumeProperty.SetScalarOpacity(self.scalarOpacity)
            self.volumeProperty.SetColor(self.color)
            
            # Create volume if not exists
            if self.volume is None:
                self.volume = vtk.vtkVolume()
                
            self.volume.SetMapper(self.volumeMapper)
            self.volume.SetProperty(self.volumeProperty)

            return self.volume
            
        except Exception as e:
            print(f"Error creating volume object: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_dicom_dataset(self, dicom_files):
        try:
            self.reset_before_load()
            if not dicom_files:
                print("No DICOM files provided")
                return False
            
            self.reset_before_load()
            
            directory_path = os.path.dirname(dicom_files[0])
            print(f"Loading DICOM from directory: {directory_path}")
            
            # Create new reader
            self.reader = vtk.vtkDICOMImageReader()
            self.reader.SetDirectoryName(directory_path)
            self.reader.SetDataByteOrderToLittleEndian()
            
            self.reader.Update()
            
            if self.reader.GetErrorCode() != 0:
                print(f"DICOM reader error code: {self.reader.GetErrorCode()}")
                return False
            
            image_data = self.reader.GetOutput()
            if not image_data:
                print("No image data received from DICOM reader")
                return False
                
            print(f"DICOM dataset loaded successfully!")
            
            extent = self.reader.GetDataExtent()
            width = extent[1] - extent[0] + 1
            height = extent[3] - extent[2] + 1
            depth = extent[5] - extent[4] + 1
            
            print(f"Dimensions: {width} x {height} x {depth}")
            print(f"File count: {len(dicom_files)}")
            
            dataset_key = os.path.abspath(directory_path)   # unique key
            self.loaded_datasets[dataset_key] = {
                'path': directory_path,
                'reader': self.reader,
                'file_count': len(dicom_files),
                'dimensions': (width, height, depth)
            }
            self.current_dataset = dataset_key
                        
            self.update_visualization()
            
            return True
            
        except Exception as e:
            print(f"Error loading DICOM dataset: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_visualization(self):
        try:
            if not self.renderer or not self.reader:
                print("Renderer or reader not initialized")
                return
            
            # Remove existing volume if exists
            if self.volume:
                self.renderer.RemoveVolume(self.volume)
                self.volume = None
            
            # Remove existing axes if exists
            if self.coordinate_axes:
                self.renderer.RemoveActor(self.coordinate_axes)
                self.coordinate_axes = None
            
            volume = self.create_object()
            if volume:
                self.volume = volume
                self.renderer.AddVolume(volume)
                
                # Add coordinate axes
                self.coordinate_axes = self.create_simple_coordinate_system()
                self.renderer.AddActor(self.coordinate_axes)
                
                self.renderer.ResetCamera()
                self.initial_camera_position = self.get_camera_state()
                
                if self.render_window:
                    self.render_window.Render()
                print("Visualization updated successfully")
                
                self.add_slicing_planes()
            else:
                print("Failed to create volume object")
                
        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()

    def create_fixed_slice(self, orientation="axial"):
        if self.reader is None:
            print("create_fixed_slice: reader not ready")
            return None, None, None
        
        self.reader.Update()
        image_data = self.reader.GetOutput()
        if not image_data:
            print("create_fixed_slice: No image data from reader")
            return None, None, None

        reslice = vtk.vtkImageReslice()
        reslice.SetInputConnection(self.reader.GetOutputPort())
        reslice.SetInterpolationModeToLinear()
        reslice.SetOutputDimensionality(2)

        extent = image_data.GetExtent()
        spacing = image_data.GetSpacing()
        origin = image_data.GetOrigin()
        
        axial = vtk.vtkMatrix4x4()
        axial.DeepCopy([1,0,0,0,
                        0,1,0,0,
                        0,0,1,0,
                        0,0,0,1])

        coronal = vtk.vtkMatrix4x4()
        coronal.DeepCopy([1,0,0,0,
                        0,0,1,0,
                        0,1,0,0,
                        0,0,0,1])

        sagittal = vtk.vtkMatrix4x4()
        sagittal.DeepCopy([0,0,1,0,
                            0,1,0,0,
                            1,0,0,0,
                            0,0,0,1])

        if orientation == "axial":
            reslice.SetResliceAxes(axial)
            slice_index = (extent[4] + extent[5]) // 2
            world_pos = origin[2] + slice_index * spacing[2]
            axial.SetElement(2, 3, world_pos)
            
        elif orientation == "coronal":
            reslice.SetResliceAxes(coronal)
            slice_index = (extent[2] + extent[3]) // 2
            world_pos = origin[1] + slice_index * spacing[1]
            coronal.SetElement(1, 3, world_pos)
            
        else:
            reslice.SetResliceAxes(sagittal)
            slice_index = (extent[0] + extent[1]) // 2
            world_pos = origin[0] + slice_index * spacing[0]
            sagittal.SetElement(0, 3, world_pos)

        mapper = vtk.vtkImageSliceMapper()
        mapper.SetInputConnection(reslice.GetOutputPort())

        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        
        # Set window/level
        actor.GetProperty().SetColorWindow(self.window_value)
        actor.GetProperty().SetColorLevel(self.level_value)

        return reslice, mapper, actor

    def add_slicing_planes(self):
        try:
            # Clear existing slice planes
            self.slice_planes.clear()
            
            if self.reader is None:
                print("Cannot add slices: reader not available")
                return
            
            self.reader.Update()
            image_data = self.reader.GetOutput()
            
            if not image_data:
                print("Cannot add slices: no image data")
                return
            
            self.imageData = image_data
            
            print(f"Creating slices from image data with dimensions: {image_data.GetDimensions()}")
            
            # Create slices for all orientations
            orientations = ["axial", "coronal", "sagittal"]
            for orientation in orientations:
                r, m, a = self.create_fixed_slice(orientation)
                if a:
                    self.slice_planes[orientation] = (r, m, a)

            print("Slicing planes created for preview:", list(self.slice_planes.keys()))

        except Exception as e:
            print("Error creating slicing planes:", e)
            import traceback
            traceback.print_exc()
        
    def update_slice(self, axis, value):
        if axis not in self.slice_planes:
            return

        reslice, mapper, actor = self.slice_planes[axis]
        if reslice is None or self.reader is None:
            return

        self.reader.Update()
        image_data = self.reader.GetOutput()
        if not image_data:
            return

        extent = image_data.GetExtent()
        spacing = image_data.GetSpacing()
        origin = image_data.GetOrigin()

        if axis == "axial":
            min_s, max_s = extent[4], extent[5]
            translate_axis = 2
        elif axis == "coronal":
            min_s, max_s = extent[2], extent[3]
            translate_axis = 1
        else:
            min_s, max_s = extent[0], extent[1]
            translate_axis = 0

        if max_s <= min_s:
            print("update_slice: invalid extent for", axis, extent)
            return

        slice_index = min_s + int((value / 1000.0) * (max_s - min_s))
        slice_index = max(min_s, min(slice_index, max_s))  # Clamp value
        slice_world_pos = origin[translate_axis] + slice_index * spacing[translate_axis]

        old_matrix = reslice.GetResliceAxes()
        if old_matrix is None:
            mat = vtk.vtkMatrix4x4()
        else:
            mat = vtk.vtkMatrix4x4()
            mat.DeepCopy(old_matrix)

        mat.SetElement(translate_axis, 3, slice_world_pos)
        reslice.SetResliceAxes(mat)
        reslice.Modified()
        
        print(f"{axis.upper()} slice updated to index {slice_index}/{max_s}, world pos: {slice_world_pos:.1f}")

        # Update all views
        self.update_all_slice_views()
        
        if self.render_window:
            self.render_window.Render()
    
    def update_all_slice_views(self):
        """Update slice views for both main and slice view modes"""
        if self.main_window:
            if self.main_window.center_stack.currentIndex() == 1:  # Slice view active
                self.update_slice_view_widgets()
            else:  # Main view active
                self.update_preview_widgets()
    
    def update_slice_view_widgets(self):
        """Update the large slice view widgets"""
        if not self.main_window:
            return
        
        try:
            for axis in ['axial', 'coronal', 'sagittal']:
                if axis not in self.slice_planes:
                    continue
                    
                reslice, mapper, actor = self.slice_planes[axis]
                if not reslice:
                    continue
                
                # Get the appropriate widget
                if axis == "axial":
                    widget = self.main_window.axial_slice_view
                elif axis == "coronal":
                    widget = self.main_window.coronal_slice_view
                else:
                    widget = self.main_window.sagittal_slice_view
                
                if not widget:
                    continue
                
                ren_win = widget.GetRenderWindow()
                if not ren_win:
                    continue
                
                renderers = ren_win.GetRenderers()
                if not renderers or renderers.GetNumberOfItems() == 0:
                    ren = vtk.vtkRenderer()
                    ren.SetBackground(0.1, 0.1, 0.1)
                    ren_win.AddRenderer(ren)
                    inter = ren_win.GetInteractor()
                    style = vtk.vtkInteractorStyleImage()
                    inter.SetInteractorStyle(style)
                    inter.Initialize()
                else:
                    ren = renderers.GetFirstRenderer()
                
                # Clear old actors
                ren.RemoveAllViewProps()
                
                # Create new slice actor
                preview_mapper = vtk.vtkImageSliceMapper()
                preview_mapper.SetInputConnection(reslice.GetOutputPort())
                
                preview_actor = vtk.vtkImageSlice()
                preview_actor.SetMapper(preview_mapper)
                preview_actor.GetProperty().SetColorWindow(self.window_value)
                preview_actor.GetProperty().SetColorLevel(self.level_value)
                
                ren.AddViewProp(preview_actor)
                ren.ResetCamera()
                ren_win.Render()
                widget.update()
                
        except Exception as e:
            print(f"Error updating slice view widgets: {e}")
            import traceback
            traceback.print_exc()

    def update_preview_widgets(self, axis=None):
        """Update the bottom preview widgets (for main view)"""
        if not self.main_window:
            return
        
        if axis is None:
            for ax in ['axial', 'coronal', 'sagittal']:
                self._update_single_preview(ax)
        else:
            self._update_single_preview(axis)
    
    def _update_single_preview(self, axis):
        if axis not in self.slice_planes:
            return
        
        reslice, mapper, actor = self.slice_planes[axis]
        if not reslice:
            return
        
        try:
            if axis == "axial":
                widget = self.main_window.axial_preview
            elif axis == "coronal":
                widget = self.main_window.coronal_preview
            else:
                widget = self.main_window.sagittal_preview
            
            if not widget:
                return
            
            ren_win = widget.GetRenderWindow()
            if not ren_win:
                return
            
            renderers = ren_win.GetRenderers()
            if not renderers or renderers.GetNumberOfItems() == 0:
                ren = vtk.vtkRenderer()
                ren.SetBackground(0.1, 0.1, 0.1)
                ren_win.AddRenderer(ren)
            else:
                ren = renderers.GetFirstRenderer()
            
            # Clear old props
            ren.RemoveAllViewProps()
            
            # Create new slice actor
            preview_mapper = vtk.vtkImageSliceMapper()
            preview_mapper.SetInputConnection(reslice.GetOutputPort())
            
            preview_actor = vtk.vtkImageSlice()
            preview_actor.SetMapper(preview_mapper)
            preview_actor.GetProperty().SetColorWindow(self.window_value)
            preview_actor.GetProperty().SetColorLevel(self.level_value)
            
            ren.AddViewProp(preview_actor)
            ren.ResetCamera()
            ren_win.Render()
            widget.update()
            
        except Exception as e:
            print(f"Error updating {axis} preview widget: {e}")

    def set_shading_preset(self, preset):
        if not self.volumeProperty:
            print("No volume loaded — cannot apply shading.")
            return

        vp = self.volumeProperty

        # Enable shading globally
        vp.ShadeOn()

        if preset == "phong":
            vp.SetAmbient(0.2)
            vp.SetDiffuse(0.8)
            vp.SetSpecular(0.5)
            vp.SetSpecularPower(20)
            print("Applied: Phong shading")

        elif preset == "xray":
            vp.SetAmbient(0.9)
            vp.SetDiffuse(0.1)
            vp.SetSpecular(0.0)
            vp.SetSpecularPower(1)
            print("Applied: X-Ray mode")

        elif preset == "glow":
            vp.SetAmbient(0.7)
            vp.SetDiffuse(0.3)
            vp.SetSpecular(1.0)
            vp.SetSpecularPower(80)
            print("Applied: Glow mode")

        elif preset == "depth":
            vp.SetAmbient(0.4)
            vp.SetDiffuse(0.6)
            vp.SetSpecular(0.2)
            vp.SetSpecularPower(10)
            print("Applied: Depth-enhanced shading")

        # Re-render
        if self.render_window:
            self.render_window.Render()

    def update_opacity_transfer_function(self):
        """Update the opacity transfer function for volume rendering"""
        if not self.reader:
            return
            
        self.reader.Update()
        image_data = self.reader.GetOutput()
        if not image_data:
            return
            
        scalar_range = image_data.GetScalarRange()
        min_val, max_val = scalar_range
        
        # Clear existing points
        self.opacity_function.RemoveAllPoints()
        
        # Create custom opacity function using window_value and level_value
        self.opacity_function.AddPoint(min_val, 0.0)
        self.opacity_function.AddPoint(self.level_value - self.window_value/2, 0.0)
        self.opacity_function.AddPoint(self.level_value, 0.2)
        self.opacity_function.AddPoint(self.level_value + self.window_value/2, 0.8)
        self.opacity_function.AddPoint(max_val, 1.0)
        
        # Update volume property
        if self.volumeProperty:
            self.volumeProperty.SetScalarOpacity(self.opacity_function)
            
        # Trigger re-render
        if self.render_window:
            self.render_window.Render()
    
    def save_modified_model(self, filepath):
        """Save the current modified model state"""
        try:
            if not self.reader:
                print("No model loaded to save")
                return False
            
            # Collect all state information
            state = {
                'window_value': self.window_value,
                'level_value': self.level_value,
                'opacity_points': [],
                'color_points': [],
                'slice_positions': {},
                'measurements': [],
                'camera_state': self.get_camera_state()
            }
            
            # Save opacity transfer function points
            opacity_pts = self.opacity_function.GetDataPointer()
            if opacity_pts:
                for i in range(self.opacity_function.GetSize()):
                    x = self.opacity_function.GetDataValue(i, 0)
                    y = self.opacity_function.GetDataValue(i, 1)
                    state['opacity_points'].append((x, y))
            
            # Save color transfer function points
            color_pts = self.color_function.GetDataPointer()
            if color_pts:
                for i in range(self.color_function.GetSize()):
                    x = self.color_function.GetDataValue(i, 0)
                    r = self.color_function.GetDataValue(i, 1)
                    g = self.color_function.GetDataValue(i, 2)
                    b = self.color_function.GetDataValue(i, 3)
                    state['color_points'].append((x, r, g, b))
            
            # Save slice positions
            if self.reader:
                self.reader.Update()
                image_data = self.reader.GetOutput()
                if image_data:
                    for orientation in ['axial', 'coronal', 'sagittal']:
                        if orientation in self.slice_planes:
                            reslice, mapper, actor = self.slice_planes[orientation]
                            if reslice:
                                matrix = reslice.GetResliceAxes()
                                if matrix:
                                    state['slice_positions'][orientation] = {
                                        'matrix': [matrix.GetElement(i, j) for i in range(4) for j in range(4)]
                                    }
            
            # Save measurements
            for i in range(0, len(self.measurement_points), 2):
                if i + 1 < len(self.measurement_points):
                    state['measurements'].append({
                        'point1': self.measurement_points[i],
                        'point2': self.measurement_points[i + 1],
                        'distance_mm': self.calculate_distance() if i == 0 else 0  # Simplified
                    })
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            print(f"Model state saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_volume_color(self, rgb):
        if not self.volume or not self.imageData:
            print("No volume to recolor")
            return

        r, g, b = rgb

        # Create new transfer function
        color_tf = vtk.vtkColorTransferFunction()
        scalar_range = self.imageData.GetScalarRange()
        low, high = scalar_range

        # Apply color uniformly but follow density
        color_tf.AddRGBPoint(low,   r * 0.3, g * 0.3, b * 0.3)
        color_tf.AddRGBPoint(high,  r,       g,       b)

        # Replace the existing transfer function
        self.volumeProperty.SetColor(color_tf)

        # Redraw
        if self.render_window:
            self.render_window.Render()

        print(f"Volume color updated to: {rgb}")

    def load_modified_model(self, filepath, dicom_directory):
        """Load a previously saved model state"""
        try:
            # First load the DICOM dataset
            success = self.load_dicom_dataset([dicom_directory])
            if not success:
                return False
            
            # Load saved state
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore window/level
            self.window_value = state.get('window_value', 400)
            self.level_value = state.get('level_value', 40)
            
            # Restore opacity function
            self.opacity_function.RemoveAllPoints()
            for x, y in state.get('opacity_points', []):
                self.opacity_function.AddPoint(x, y)
            
            # Restore color function
            self.color_function.RemoveAllPoints()
            for x, r, g, b in state.get('color_points', []):
                self.color_function.AddRGBPoint(x, r, g, b)
            
            # Update volume properties
            if self.volumeProperty:
                self.volumeProperty.SetScalarOpacity(self.opacity_function)
                self.volumeProperty.SetColor(self.color_function)
            
            # Restore camera state
            if 'camera_state' in state and state['camera_state']:
                self.set_camera_state(state['camera_state'])
            
            # Restore measurements
            if 'measurements' in state:
                for measurement in state['measurements']:
                    self.measurement_points.append(measurement['point1'])
                    self.measurement_points.append(measurement['point2'])
                    self.create_measurement_point_marker(measurement['point1'])
                    self.create_measurement_point_marker(measurement['point2'])
                    self.create_measurement_line_improved(
                        measurement['point1'], 
                        measurement['point2'],
                        measurement.get('distance_mm', 0),
                        0
                    )
            
            # Update visualization
            self.update_visualization()
            
            print(f"Model state loaded from: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def export_model_as_image(self, filepath, format='PNG', size=(800, 600)):
        """Export current view as an image"""
        try:
            if not self.render_window:
                return False
            
            # Create window to image filter
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.render_window)
            
            if format.upper() == 'PNG':
                writer = vtk.vtkPNGWriter()
            elif format.upper() == 'JPEG':
                writer = vtk.vtkJPEGWriter()
            else:
                writer = vtk.vtkPNGWriter()
            
            writer.SetFileName(filepath)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
            
            print(f"Image saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error exporting image: {e}")
            return False
    
    def update_color_transfer_function(self):
        """Update the color transfer function"""
        if not self.reader:
            return
            
        self.reader.Update()
        image_data = self.reader.GetOutput()
        if not image_data:
            return
            
        scalar_range = image_data.GetScalarRange()
        min_val, max_val = scalar_range
        
        # Clear existing points
        self.color_function.RemoveAllPoints()
        
        # Create color gradient
        self.color_function.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
        self.color_function.AddRGBPoint(self.level_value - self.window_value/3, 0.3, 0.3, 0.3)
        self.color_function.AddRGBPoint(self.level_value, 0.7, 0.7, 0.7)
        self.color_function.AddRGBPoint(self.level_value + self.window_value/3, 1.0, 1.0, 1.0)
        self.color_function.AddRGBPoint(max_val, 1.0, 1.0, 1.0)
        
        # Update volume property
        if self.volumeProperty:
            self.volumeProperty.SetColor(self.color_function)
            
        if self.render_window:
            self.render_window.Render()
    
    def set_window_level(self, window, level):
        """Set window/level for adjusting contrast/brightness"""
        self.window_value = window
        self.level_value = level
        
        # Update both opacity and color functions
        self.update_opacity_transfer_function()
        self.update_color_transfer_function()
        
        # Also update slice previews if they exist
        if self.main_window:
            for axis in ['axial', 'coronal', 'sagittal']:
                if axis in self.slice_planes:
                    reslice, mapper, actor = self.slice_planes[axis]
                    if actor:
                        actor.GetProperty().SetColorWindow(window)
                        actor.GetProperty().SetColorLevel(level)
            
            # Update all views
            self.update_all_slice_views()
            
        print(f"Window/Level set to: {window}/{level}")
        
    def adjust_opacity_range(self, inner_opacity=0.1, outer_opacity=1.0):
        """Adjust opacity for inner vs outer structures"""
        if not self.reader:
            return
            
        self.reader.Update()
        image_data = self.reader.GetOutput()
        if not image_data:
            return
            
        scalar_range = image_data.GetScalarRange()
        min_val, max_val = scalar_range
        
        # Clear and set new opacity function
        self.opacity_function.RemoveAllPoints()
        
        mid_point = (self.level_value - self.window_value/2 + 
                    self.level_value + self.window_value/2) / 2
        
        # Inner structures (lower values) get lower opacity
        self.opacity_function.AddPoint(min_val, inner_opacity * 0.1)
        self.opacity_function.AddPoint(self.level_value - self.window_value/2, inner_opacity * 0.3)
        self.opacity_function.AddPoint(mid_point, inner_opacity * 0.5)
        
        # Outer structures (higher values) get higher opacity
        self.opacity_function.AddPoint(self.level_value + self.window_value/2, outer_opacity * 0.7)
        self.opacity_function.AddPoint(max_val, outer_opacity)
        
        if self.volumeProperty:
            self.volumeProperty.SetScalarOpacity(self.opacity_function)
            
        if self.render_window:
            self.render_window.Render()

    def start_measurement(self):
        self.is_measuring = True
        self.measurement_points = []
        print("Measurement mode: Click on two points to measure distance")

    def stop_measurement(self):
        self.is_measuring = False
        self.measurement_points = []

    def add_measurement_point(self, point):
        if not self.is_measuring:
            return
        
        print(f"Adding measurement point {len(self.measurement_points) + 1}: {point}")
        
        self.measurement_points.append(point)
        self.create_measurement_point_marker(point)
        
        if len(self.measurement_points) == 2:
            print(f"Two points collected, calculating distance...")
            distance_mm, distance_pixels = self.calculate_distance()
            
            if self.main_window:
                self.main_window.measurement_output.setText(
                    f"Distance:\n{distance_mm:.1f} mm\n{distance_pixels:.0f} pixels"
                )

    def calculate_distance(self):
        if len(self.measurement_points) != 2:
            print(f"Error: Need exactly 2 points, got {len(self.measurement_points)}")
            return 0, 0
            
        point1 = self.measurement_points[0]
        point2 = self.measurement_points[1]
        
        print(f"Point 1: {point1}")
        print(f"Point 2: {point2}")
        
        distance_mm = ((point2[0] - point1[0])**2 + 
                    (point2[1] - point1[1])**2 + 
                    (point2[2] - point1[2])**2)**0.5
        
        if self.reader:
            try:
                self.reader.Update()
                spacing = self.reader.GetDataSpacing()
                distance_pixels = (
                    ((point2[0] - point1[0]) / spacing[0])**2 +
                    ((point2[1] - point1[1]) / spacing[1])**2 +
                    ((point2[2] - point1[2]) / spacing[2])**2
                )**0.5
                print(f"Spacing: {spacing}")
            except Exception as e:
                print(f"Error calculating pixel distance: {e}")
                distance_pixels = distance_mm
        else:
            distance_pixels = distance_mm
            print("No reader found, using fallback")
        
        print(f"Distance: {distance_mm:.1f} mm, {distance_pixels:.0f} pixels")
        
        self.create_measurement_line_improved(point1, point2, distance_mm, distance_pixels)
        
        return distance_mm, distance_pixels

    def create_measurement_point_marker(self, position, color=(1,1,0)):
        try:
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(position)
            sphere_source.SetRadius(3.0)
            sphere_source.SetPhiResolution(16)
            sphere_source.SetThetaResolution(16)
            
            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
            
            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.GetProperty().SetColor(color)
            sphere_actor.GetProperty().SetOpacity(0.9)
            sphere_actor.GetProperty().SetAmbient(0.5)
            sphere_actor.GetProperty().SetDiffuse(0.5)
            sphere_actor.GetProperty().SetSpecular(0.5)
            
            self.measurement_actors.append(sphere_actor)
            
            if self.renderer:
                self.renderer.AddActor(sphere_actor)
                if self.render_window:
                    self.render_window.Render()
            
            print(f"Created measurement marker at: {position}")
            
        except Exception as e:
            print(f"Error creating point marker: {e}")
            import traceback
            traceback.print_exc()

    def create_measurement_line_improved(self, point1, point2, distance_mm, distance_pixels):
        try:
            print("Creating measurement line...")

            line = vtk.vtkLineSource()
            line.SetPoint1(point1)
            line.SetPoint2(point2)

            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(line.GetOutputPort())
            tube.SetRadius(2)
            tube.SetNumberOfSides(20)

            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInputConnection(tube.GetOutputPort())

            line_actor = vtk.vtkActor()
            line_actor.SetMapper(line_mapper)
            line_actor.GetProperty().SetColor(1, 1, 0)
            line_actor.GetProperty().SetOpacity(1.0)
            line_actor.GetProperty().SetLineWidth(4)

            text = vtk.vtkBillboardTextActor3D()
            text.SetInput(f"{distance_mm:.1f} mm\n({distance_pixels:.0f} px)")

            mid = [
                (point1[0] + point2[0]) / 2,
                (point1[1] + point2[1]) / 2,
                (point1[2] + point2[2]) / 2
            ]
            text.SetPosition(mid)
            text.GetTextProperty().SetFontSize(20)
            text.GetTextProperty().SetColor(1, 1, 0)
            text.GetTextProperty().SetBold(True)
            text.GetTextProperty().SetBackgroundOpacity(0.6)
            text.GetTextProperty().SetBackgroundColor(0, 0, 0)

            self.renderer.AddActor(line_actor)
            self.renderer.AddActor(text)

            self.measurement_actors.append(line_actor)
            self.measurement_actors.append(text)

            if self.render_window:
                self.render_window.Render()

            print("Measurement line created")

        except Exception as e:
            print("Error creating measurement line:", e)
            import traceback
            traceback.print_exc()
        
    def clear_measurements(self):
        if self.renderer:
            for actor in self.measurement_actors:
                try:
                    self.renderer.RemoveActor(actor)
                except:
                    pass
        self.measurement_actors.clear()
        self.measurement_points.clear()
        
        if self.main_window:
            self.main_window.measurement_output.setText('Distance: -')
        if self.render_window:
            self.render_window.Render()
        print("All measurements cleared")

    def setup_rendering_pipeline(self, vtk_widget):
        try:
            # Ensure widget is properly initialized
            ren_win = vtk_widget.GetRenderWindow()
            ren_win.SetOffScreenRendering(0)
            
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.1)
            self.render_window = ren_win
            
            # Clear any existing renderers
            self.render_window.GetRenderers().RemoveAllItems()
            self.render_window.AddRenderer(self.renderer)
            
            self.interactor = self.render_window.GetInteractor()
            
            # Set up interactor style for measurement
            self.interactorStyle = CustomInteractorStyle(self)
            self.interactor.SetInteractorStyle(self.interactorStyle)
            
            # Initialize the interactor
            if not self.interactor.GetInitialized():
                self.interactor.Initialize()
            
            self.is_initialized = True
            print("Rendering pipeline initialized successfully")
            
        except Exception as e:
            print(f"Error setting up rendering pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    def create_scene(self):
        volume = self.create_object()
        if volume:
            self.volume = volume
            self.renderer.AddVolume(volume)
            
            axes = self.create_simple_coordinate_system()
            self.coordinate_axes = axes
            self.renderer.AddActor(axes)
            
            self.renderer.ResetCamera()
            self.initial_camera_position = self.get_camera_state()
            
            return True
        return False
        
    def get_camera_state(self):
        if self.renderer:
            camera = self.renderer.GetActiveCamera()
            return {
                'position': camera.GetPosition(),
                'focal_point': camera.GetFocalPoint(),
                'view_up': camera.GetViewUp(),
                'view_angle': camera.GetViewAngle()
            }
        return None
        
    def set_camera_state(self, state):
        if self.renderer and state:
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(state['position'])
            camera.SetFocalPoint(state['focal_point'])
            camera.SetViewUp(state['view_up'])
            camera.SetViewAngle(state['view_angle'])
        
    def render_all(self):
        if self.render_window:
            self.render_window.Render()
            if self.interactor and not self.interactor.GetInitialized():
                self.interactor.Initialize()
    
    def rotate_view(self, axis, angle):
        if self.renderer:
            camera = self.renderer.GetActiveCamera()
            
            if axis.lower() == 'x':
                camera.Roll(angle)
            elif axis.lower() == 'y':
                camera.Pitch(angle)
            elif axis.lower() == 'z':
                camera.Yaw(angle)
                
            if self.render_window:
                self.render_window.Render()

    def set_camera_view(self, view_type):
        if self.renderer:
            camera = self.renderer.GetActiveCamera()
            
            if view_type == 'front':
                camera.SetPosition(0, 0, 1)
                camera.SetFocalPoint(0, 0, 0)
                camera.SetViewUp(0, 1, 0)
            elif view_type == 'side':
                camera.SetPosition(1, 0, 0)
                camera.SetFocalPoint(0, 0, 0)
                camera.SetViewUp(0, 1, 0)
            elif view_type == 'top':
                camera.SetPosition(0, 1, 0)
                camera.SetFocalPoint(0, 0, 0)
                camera.SetViewUp(0, 0, 1)
                
            if self.render_window:
                self.render_window.Render()

    def reset_before_load(self):
        """Remove all actors/volume/slices and reset readers prior to loading a new dataset."""
        try:
            if self.renderer:
                # Remove volume
                if self.volume:
                    try:
                        self.renderer.RemoveVolume(self.volume)
                    except:
                        pass
                    self.volume = None

                # Remove coordinate axes
                if self.coordinate_axes:
                    try:
                        self.renderer.RemoveActor(self.coordinate_axes)
                    except:
                        pass
                    self.coordinate_axes = None

                # Remove measurement actors
                for a in list(self.measurement_actors):
                    try:
                        self.renderer.RemoveActor(a)
                    except:
                        pass
                self.measurement_actors.clear()

                # Remove slice plane props if they were added to this renderer
                for axis, triple in list(self.slice_planes.items()):
                    reslice, mapper, actor = triple
                    try:
                        if actor:
                            self.renderer.RemoveActor(actor)
                    except:
                        pass
                self.slice_planes.clear()

                # Force renderer to clear props & redraw
                try:
                    self.renderer.RemoveAllViewProps()
                except:
                    pass

            # Drop reader and image data
            self.reader = None
            self.imageData = None
            self.current_dataset = None

            # Render a blank frame
            if self.render_window:
                self.render_window.Render()

            print("VTK reset_before_load completed.")
        except Exception as e:
            print("Error in reset_before_load:", e)

    def reset_view(self):
        if self.renderer:
            if self.initial_camera_position:
                self.set_camera_state(self.initial_camera_position)
            else:
                self.renderer.ResetCamera()
            if self.render_window:
                self.render_window.Render()

    def start(self, vtk_widget):
        self.setup_rendering_pipeline(vtk_widget)
        self.create_scene()
        self.add_slicing_planes()
        self.render_all()

    def cleanup(self):
        """Clean up all VTK resources"""
        try:
            # Stop any ongoing rendering
            if self.render_window and self.render_window.GetInteractor():
                self.render_window.GetInteractor().TerminateApp()
            
            # Clear all actors
            if self.renderer:
                self.renderer.RemoveAllViewProps()
            
            # Clear references
            self.actors.clear()
            self.mappers.clear()
            self.sources.clear()
            self.slice_planes.clear()
            self.measurement_actors.clear()
            self.measurement_points.clear()
            
            # Reset state
            self.is_initialized = False
            self.is_rendering = False
            
            print("VTK resources cleaned up")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

class LoadModelThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, list, str)  # message, dicom_files, directory_path
    error = pyqtSignal(str)

    def __init__(self, dicom_files, directory_path):
        super().__init__()
        self.dicom_files = dicom_files
        self.directory_path = directory_path

    def run(self):
        try:
            self.progress.emit(10)
            # Simulate scanning/validation (non-VTK work)
            QThread.msleep(100)
            # If we got files, signal the main thread to perform VTK load
            if self.dicom_files:
                self.progress.emit(50)
                QThread.msleep(50)
                self.progress.emit(100)
                # Emit finished with the file list and path — main thread will do VTK load
                self.finished.emit("Ready to load", self.dicom_files, self.directory_path)
            else:
                self.error.emit("No DICOM files found in worker")
        except Exception as e:
            self.error.emit(str(e))

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, vtk_app):
        super().__init__()
        self.vtk_app = vtk_app
        self.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
    
    def on_left_button_press(self, obj, event):
        if self.vtk_app.is_measuring:
            # Use volume picker and main renderer for measurements
            interactor = self.vtk_app.interactor
            click_pos = interactor.GetEventPosition()
            picker = vtk.vtkVolumePicker()
            picker.SetTolerance(1e-6)
            picker.Pick(click_pos[0], click_pos[1], 0, self.vtk_app.renderer)
            if picker.GetCellId() != -1:
                pos = picker.GetPickPosition()
                self.vtk_app.create_measurement_point_marker(pos)
                self.vtk_app.add_measurement_point(pos)
            return
        self.OnLeftButtonDown()

class AnimatedSplashScreen(QWidget):
    def __init__(self, gif_path):
        super().__init__()

        # No frame, behave like splash
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.SplashScreen)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # === Build absolute path so it works no matter where you run it ===
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, gif_path)

        print("DEBUG: Loading GIF from:", full_path)

        self.movie = QMovie(full_path)

        if not self.movie.isValid():
            print("DEBUG: GIF failed to load! Showing text instead.")
            self.label.setText("SPLASH GIF NOT FOUND")
        else:
            self.label.setMovie(self.movie)
            self.movie.start()

            # Resize window to GIF size
            self.resize(self.movie.frameRect().size())
            print("DEBUG: GIF size:", self.movie.frameRect().size())

        # Center on screen
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2
            )



class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Advanced DICOM Medical Visualizer - Dual View Mode")
        self.loaded_models = {}
        self.current_model = None
        self.preview_slices = {}
        self.is_measuring = False
        
        self.apply_dark_theme()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create menus
        self.create_menus()
        
        # Create top view toggle buttons
        self.create_view_toggle_buttons(main_layout)
        
        # Create main content area with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel for model list
        self.create_models_panel(splitter)
        
        # Create central stacked widget for Main View / Slice View
        self.center_stack = QStackedWidget()
        
        # MAIN VIEW (Index 0)
        self.main_view_container = QWidget()
        main_view_layout = QVBoxLayout()
        self.main_view_container.setLayout(main_view_layout)
        
        # Main VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self.main_view_container)
        main_view_layout.addWidget(self.vtk_widget)
        
        # Bottom preview panel for main view
        self.create_bottom_preview_panel(main_view_layout)
        
        self.center_stack.addWidget(self.main_view_container)
        
        # SLICE VIEW (Index 1)
        self.slice_view_container = QWidget()
        slice_view_layout = QVBoxLayout()
        self.slice_view_container.setLayout(slice_view_layout)
        
        # Top row: Axial and Sagittal (equal halves)
        top_row = QWidget()
        top_layout = QHBoxLayout()
        top_row.setLayout(top_layout)
        
        # Axial slice view (left half)
        axial_container = QWidget()
        axial_vbox = QVBoxLayout()
        axial_container.setLayout(axial_vbox)
        axial_label = QLabel("AXIAL SLICE")
        axial_label.setAlignment(Qt.AlignCenter)
        axial_label.setStyleSheet("font-weight: bold; color: red; font-size: 14px; padding: 5px;")
        axial_vbox.addWidget(axial_label)
        self.axial_slice_view = QVTKRenderWindowInteractor(axial_container)
        self.axial_slice_view.setMinimumSize(400, 350)
        axial_vbox.addWidget(self.axial_slice_view)
        top_layout.addWidget(axial_container, 1)
        
        # Sagittal slice view (right half)
        sagittal_container = QWidget()
        sagittal_vbox = QVBoxLayout()
        sagittal_container.setLayout(sagittal_vbox)
        sagittal_label = QLabel("SAGITTAL SLICE")
        sagittal_label.setAlignment(Qt.AlignCenter)
        sagittal_label.setStyleSheet("font-weight: bold; color: green; font-size: 14px; padding: 5px;")
        sagittal_vbox.addWidget(sagittal_label)
        self.sagittal_slice_view = QVTKRenderWindowInteractor(sagittal_container)
        self.sagittal_slice_view.setMinimumSize(400, 350)
        sagittal_vbox.addWidget(self.sagittal_slice_view)
        top_layout.addWidget(sagittal_container, 1)
        
        # Bottom row: Coronal (full width, maximized)
        bottom_row = QWidget()
        bottom_layout = QVBoxLayout()
        bottom_row.setLayout(bottom_layout)
        
        coronal_container = QWidget()
        coronal_vbox = QVBoxLayout()
        coronal_container.setLayout(coronal_vbox)
        coronal_label = QLabel("CORONAL SLICE (Maximized)")
        coronal_label.setAlignment(Qt.AlignCenter)
        coronal_label.setStyleSheet("font-weight: bold; color: blue; font-size: 14px; padding: 5px;")
        coronal_vbox.addWidget(coronal_label)
        self.coronal_slice_view = QVTKRenderWindowInteractor(coronal_container)
        self.coronal_slice_view.setMinimumSize(800, 400)
        coronal_vbox.addWidget(self.coronal_slice_view)
        bottom_layout.addWidget(coronal_container)
        
        # Set stretch factors to give more space to coronal
        slice_view_layout.addWidget(top_row, 1)
        slice_view_layout.addWidget(bottom_row, 2)
        
        self.center_stack.addWidget(self.slice_view_container)
        
        splitter.addWidget(self.center_stack)
        
        # Create right control panel
        self.create_control_panel(splitter)
        
        splitter.setSizes([200, 800, 300])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Initialize myVTK
        self.vtk_app = myVTK()
        self.vtk_app.main_window = self
        self.vtk_app.start(self.vtk_widget)
        
        # Start in Main View
        self.set_main_view()
        #self.showFullScreen()
        
    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QMenuBar {
                background-color: #2b2b2b;
                color: white;
                border: none;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #404040;
            }
            QMenu {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #555;
            }
            QMenu::item {
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #404040;
            }
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #555;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QListWidget {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #4CAF50;
                color: white;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QLabel {
                background-color: transparent;
                color: white;
            }
            QSlider::groove:horizontal {
                background: #404040;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
        """)
    
    def create_view_toggle_buttons(self, parent_layout):
        container = QWidget()
        layout = QHBoxLayout()
        container.setLayout(layout)
        
        layout.addStretch()
        
        self.main_view_btn = QPushButton('Main View')
        self.main_view_btn.setFixedHeight(40)
        self.main_view_btn.setMinimumWidth(120)
        self.main_view_btn.clicked.connect(self.set_main_view)
        
        self.slice_view_btn = QPushButton('Slice View')
        self.slice_view_btn.setFixedHeight(40)
        self.slice_view_btn.setMinimumWidth(120)
        self.slice_view_btn.clicked.connect(self.set_slice_view)
        
        layout.addWidget(self.main_view_btn)
        layout.addWidget(self.slice_view_btn)
        layout.addStretch()
        
        parent_layout.addWidget(container)
    
    def create_menus(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        load_action = QAction('Open DICOM', self)
        load_action.setShortcut('Ctrl+O')
        load_action.setStatusTip('Open DICOM dataset')
        load_action.triggered.connect(self.load_dicom_model)
        file_menu.addAction(load_action)
        
        # NEW: Save modified model
        save_action = QAction('Save Model State', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_model_state)
        file_menu.addAction(save_action)
        
        # NEW: Load modified model
        load_saved_action = QAction('Load Model State', self)
        load_saved_action.setShortcut('Ctrl+L')
        load_saved_action.triggered.connect(self.load_model_state)
        file_menu.addAction(load_saved_action)
        
        # NEW: Export as image
        export_action = QAction('Export View as Image', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_view_image)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu('View')
        main_view_action = QAction('Main View', self)
        main_view_action.setShortcut('Ctrl+1')
        main_view_action.triggered.connect(self.set_main_view)
        view_menu.addAction(main_view_action)
        
        slice_view_action = QAction('Slice View', self)
        slice_view_action.setShortcut('Ctrl+2')
        slice_view_action.triggered.connect(self.set_slice_view)
        view_menu.addAction(slice_view_action)
        
        view_menu.addSeparator()
        reset_view_action = QAction('Reset View', self)
        reset_view_action.setShortcut('Ctrl+R')
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        
        help_menu = menubar.addMenu('Help')
        help_action = QAction('Help Contents', self)
        help_action.setShortcut('F1')
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
    
    def create_models_panel(self, parent_splitter):
        models_panel = QWidget()
        models_layout = QVBoxLayout()
        models_panel.setLayout(models_layout)
        
        models_title = QLabel("Loaded Models")
        models_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #4CAF50;
                padding: 10px;
                border-bottom: 2px solid #4CAF50;
            }
        """)
        models_layout.addWidget(models_title)
        
        self.models_list = QListWidget()
        self.models_list.itemClicked.connect(self.on_model_selected)
        models_layout.addWidget(self.models_list)
        
        model_actions_layout = QHBoxLayout()
        info_btn = QPushButton("Model Info")
        info_btn.clicked.connect(self.show_model_info)
        remove_btn = QPushButton("Remove Model")
        remove_btn.clicked.connect(self.remove_model)
        model_actions_layout.addWidget(info_btn)
        model_actions_layout.addWidget(remove_btn)
        models_layout.addLayout(model_actions_layout)
        
        parent_splitter.addWidget(models_panel)

        # --- Color Palette Section ---
        self.color_label = QLabel("Model Color:")
        self.color_label.setStyleSheet("font-weight: bold; margin-top: 10px;")

        self.color_button = QPushButton()
        self.color_button.setFixedSize(40, 20)
        self.color_button.setStyleSheet("background-color: #FFFFFF; border: 1px solid #555;")
        self.color_button.clicked.connect(self.open_color_dialog)

        self.color_layout = QHBoxLayout()
        self.color_layout.addWidget(self.color_label)
        self.color_layout.addWidget(self.color_button)

        # FIX: Add to models_layout, not right_panel_layout
        models_layout.addLayout(self.color_layout)
        
        # --- Shading Presets ---
        shading_title = QLabel("Shading Presets:")
        shading_title.setStyleSheet("font-weight: bold; margin-top: 10px;")

        # Create buttons
        phong_btn = QPushButton("Phong")
        phong_btn.clicked.connect(lambda: self.vtk_app.set_shading_preset("phong"))

        xray_btn = QPushButton("X-Ray")
        xray_btn.clicked.connect(lambda: self.vtk_app.set_shading_preset("xray"))

        glow_btn = QPushButton("Glow")
        glow_btn.clicked.connect(lambda: self.vtk_app.set_shading_preset("glow"))

        depth_btn = QPushButton("Depth")
        depth_btn.clicked.connect(lambda: self.vtk_app.set_shading_preset("depth"))

        # Create a grid layout (2 rows × 2 columns)
        shade_grid = QGridLayout()
        shade_grid.addWidget(phong_btn, 0, 0)
        shade_grid.addWidget(xray_btn, 0, 1)
        shade_grid.addWidget(glow_btn, 1, 0)
        shade_grid.addWidget(depth_btn, 1, 1)

        # Add to main model panel layout
        models_layout.addWidget(shading_title)
        models_layout.addLayout(shade_grid)

    def create_control_panel(self, parent_splitter):
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        
        title = QLabel("Controls")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-weight: bold; 
                font-size: 14px;
                margin: 8px;
                color: #4CAF50;
                border-bottom: 1px solid #4CAF50;
                padding-bottom: 3px;
            }
        """)
        control_layout.addWidget(title)
        
        # Measurement section
        measurement_section = QLabel("Measurement Tools")
        measurement_section.setStyleSheet("font-weight: bold; margin-top: 15px; color: #FF9800;")
        control_layout.addWidget(measurement_section)
        
        measurement_layout = QHBoxLayout()
        self.measure_btn = QPushButton("Start Measure")
        self.measure_btn.clicked.connect(self.toggle_measurement)
        self.measure_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 6px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        clear_measure_btn = QPushButton("Clear")
        clear_measure_btn.clicked.connect(self.clear_measurements)
        clear_measure_btn.setStyleSheet("font-size: 11px; padding: 6px; margin: 2px;")
        measurement_layout.addWidget(self.measure_btn)
        measurement_layout.addWidget(clear_measure_btn)
        control_layout.addLayout(measurement_layout)
        
        measure_info = QLabel("Click two points to measure")
        measure_info.setStyleSheet("font-size: 10px; color: #AAAAAA; margin: 5px;")
        measure_info.setWordWrap(True)
        control_layout.addWidget(measure_info)
        
        self.measurement_output = QLabel("Distance: -")
        self.measurement_output.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                padding: 6px;
                border-radius: 4px;
                font-size: 12px;
                margin-top: 5px;
                color: #FFEB3B;
                border: 1px solid #555;
            }
        """)
        self.measurement_output.setWordWrap(True)
        control_layout.addWidget(self.measurement_output)
        
        # Slice controls
        slice_section = QLabel("2D Slice Controls")
        slice_section.setStyleSheet("font-weight:bold; color:#FFD700; margin-top: 15px;")
        control_layout.addWidget(slice_section)
        
        slice_info = QLabel("Works in both Main and Slice views")
        slice_info.setStyleSheet("font-size: 10px; color: #AAAAAA; margin: 2px;")
        slice_info.setWordWrap(True)
        control_layout.addWidget(slice_info)
        
        control_layout.addWidget(QLabel("Axial Slice"))
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setRange(0, 1000)
        self.axial_slider.setValue(500)
        self.axial_slider.valueChanged.connect(lambda v: self.vtk_app.update_slice("axial", v))
        control_layout.addWidget(self.axial_slider)
        
        control_layout.addWidget(QLabel("Coronal Slice"))
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setRange(0, 1000)
        self.coronal_slider.setValue(500)
        self.coronal_slider.valueChanged.connect(lambda v: self.vtk_app.update_slice("coronal", v))
        control_layout.addWidget(self.coronal_slider)
        
        control_layout.addWidget(QLabel("Sagittal Slice"))
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setRange(0, 1000)
        self.sagittal_slider.setValue(500)
        self.sagittal_slider.valueChanged.connect(lambda v: self.vtk_app.update_slice("sagittal", v))
        control_layout.addWidget(self.sagittal_slider)
        
        # Add after measurement section
        intensity_section = QLabel("Intensity/Opacity Controls")
        intensity_section.setStyleSheet("font-weight: bold; margin-top: 15px; color: #00BCD4;")
        control_layout.addWidget(intensity_section)
        
        # Window control
        window_layout = QHBoxLayout()
        window_label = QLabel("Window:")
        window_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        window_layout.addWidget(window_label)
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setRange(100, 2000)
        self.window_slider.setValue(400)
        self.window_slider.setTickInterval(100)
        self.window_slider.valueChanged.connect(self.update_window_level)
        window_layout.addWidget(self.window_slider)
        self.window_value_label = QLabel("400")
        self.window_value_label.setFixedWidth(40)
        window_layout.addWidget(self.window_value_label)
        control_layout.addLayout(window_layout)
        
        # Level control
        level_layout = QHBoxLayout()
        level_label = QLabel("Level:")
        level_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        level_layout.addWidget(level_label)
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(-100, 1000)
        self.level_slider.setValue(40)
        self.level_slider.setTickInterval(50)
        self.level_slider.valueChanged.connect(self.update_window_level)
        level_layout.addWidget(self.level_slider)
        self.level_value_label = QLabel("40")
        self.level_value_label.setFixedWidth(40)
        level_layout.addWidget(self.level_value_label)
        control_layout.addLayout(level_layout)
        
        # Opacity controls
        opacity_section = QLabel("Opacity Adjustment")
        opacity_section.setStyleSheet("font-weight: bold; margin-top: 10px; color: #E91E63;")
        control_layout.addWidget(opacity_section)
        
        # Inner opacity
        inner_layout = QHBoxLayout()
        inner_label = QLabel("Inner:")
        inner_label.setStyleSheet("color: #FF5722;")
        inner_layout.addWidget(inner_label)
        self.inner_opacity_slider = QSlider(Qt.Horizontal)
        self.inner_opacity_slider.setRange(0, 100)
        self.inner_opacity_slider.setValue(10)
        self.inner_opacity_slider.valueChanged.connect(self.update_opacity)
        inner_layout.addWidget(self.inner_opacity_slider)
        self.inner_value_label = QLabel("10%")
        self.inner_value_label.setFixedWidth(40)
        inner_layout.addWidget(self.inner_value_label)
        control_layout.addLayout(inner_layout)
        
        # Outer opacity
        outer_layout = QHBoxLayout()
        outer_label = QLabel("Outer:")
        outer_label.setStyleSheet("color: #9C27B0;")
        outer_layout.addWidget(outer_label)
        self.outer_opacity_slider = QSlider(Qt.Horizontal)
        self.outer_opacity_slider.setRange(0, 100)
        self.outer_opacity_slider.setValue(100)
        self.outer_opacity_slider.valueChanged.connect(self.update_opacity)
        outer_layout.addWidget(self.outer_opacity_slider)
        self.outer_value_label = QLabel("100%")
        self.outer_value_label.setFixedWidth(40)
        outer_layout.addWidget(self.outer_value_label)
        control_layout.addLayout(outer_layout)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        soft_tissue_btn = QPushButton("Soft Tissue")
        soft_tissue_btn.clicked.connect(lambda: self.set_preset('soft_tissue'))
        soft_tissue_btn.setStyleSheet("font-size: 10px; padding: 4px;")
        preset_layout.addWidget(soft_tissue_btn)
        
        bone_btn = QPushButton("Bone")
        bone_btn.clicked.connect(lambda: self.set_preset('bone'))
        bone_btn.setStyleSheet("font-size: 10px; padding: 4px;")
        preset_layout.addWidget(bone_btn)
        
        lung_btn = QPushButton("Lung")
        lung_btn.clicked.connect(lambda: self.set_preset('lung'))
        lung_btn.setStyleSheet("font-size: 10px; padding: 4px;")
        preset_layout.addWidget(lung_btn)
        
        control_layout.addLayout(preset_layout)
        
        # Current model info
        self.current_model_label = QLabel("No model loaded")
        self.current_model_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                padding: 6px;
                border-radius: 4px;
                font-size: 11px;
                margin: 5px;
                margin-top: 15px;
            }
        """)
        self.current_model_label.setWordWrap(True)
        control_layout.addWidget(self.current_model_label)
        
        # Rotation controls
        rotation_section = QLabel("Rotation Controls")
        rotation_section.setStyleSheet("font-weight: bold; margin-top: 10px;")
        control_layout.addWidget(rotation_section)
        
        # X-axis rotation
        x_layout = QHBoxLayout()
        x_label = QLabel("X:")
        x_label.setStyleSheet("color: red; font-weight: bold; min-width: 20px;")
        x_layout.addWidget(x_label)
        x_minus_btn = QPushButton("◀ -45°")
        x_minus_btn.clicked.connect(lambda: self.vtk_app.rotate_view('x', -45))
        x_minus_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        x_layout.addWidget(x_minus_btn)
        x_plus_btn = QPushButton("+45° ▶")
        x_plus_btn.clicked.connect(lambda: self.vtk_app.rotate_view('x', 45))
        x_plus_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        x_layout.addWidget(x_plus_btn)
        control_layout.addLayout(x_layout)
        
        # Y-axis rotation
        y_layout = QHBoxLayout()
        y_label = QLabel("Y:")
        y_label.setStyleSheet("color: green; font-weight: bold; min-width: 20px;")
        y_layout.addWidget(y_label)
        y_minus_btn = QPushButton("◀ -45°")
        y_minus_btn.clicked.connect(lambda: self.vtk_app.rotate_view('y', -45))
        y_minus_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        y_layout.addWidget(y_minus_btn)
        y_plus_btn = QPushButton("+45° ▶")
        y_plus_btn.clicked.connect(lambda: self.vtk_app.rotate_view('y', 45))
        y_plus_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        y_layout.addWidget(y_plus_btn)
        control_layout.addLayout(y_layout)
        
        # Z-axis rotation
        z_layout = QHBoxLayout()
        z_label = QLabel("Z:")
        z_label.setStyleSheet("color: blue; font-weight: bold; min-width: 20px;")
        z_layout.addWidget(z_label)
        z_minus_btn = QPushButton("◀ -45°")
        z_minus_btn.clicked.connect(lambda: self.vtk_app.rotate_view('z', -45))
        z_minus_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        z_layout.addWidget(z_minus_btn)
        z_plus_btn = QPushButton("+45° ▶")
        z_plus_btn.clicked.connect(lambda: self.vtk_app.rotate_view('z', 45))
        z_plus_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        z_layout.addWidget(z_plus_btn)
        control_layout.addLayout(z_layout)
        
        # Camera views
        camera_section = QLabel("Camera Views")
        camera_section.setStyleSheet("font-weight: bold; margin-top: 15px;")
        control_layout.addWidget(camera_section)
        
        camera_layout = QHBoxLayout()
        front_btn = QPushButton("Front")
        front_btn.clicked.connect(lambda: self.vtk_app.set_camera_view('front'))
        front_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        camera_layout.addWidget(front_btn)
        side_btn = QPushButton("Side")
        side_btn.clicked.connect(lambda: self.vtk_app.set_camera_view('side'))
        side_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        camera_layout.addWidget(side_btn)
        top_btn = QPushButton("Top")
        top_btn.clicked.connect(lambda: self.vtk_app.set_camera_view('top'))
        top_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        camera_layout.addWidget(top_btn)
        control_layout.addLayout(camera_layout)
        
        control_layout.addSpacing(10)
        
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        control_layout.addWidget(reset_btn)
        
        control_layout.addStretch()
        
        parent_splitter.addWidget(control_panel)
    
    def create_bottom_preview_panel(self, parent_layout):
        bottom_panel = QWidget()
        bottom_panel.setMaximumHeight(200)
        bottom_layout = QVBoxLayout()
        bottom_panel.setLayout(bottom_layout)
        
        preview_title = QLabel("2D Slice Previews")
        preview_title.setAlignment(Qt.AlignCenter)
        preview_title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 14px;
                color: #4CAF50;
                margin: 5px;
                border-bottom: 1px solid #4CAF50;
                padding-bottom: 3px;
            }
        """)
        bottom_layout.addWidget(preview_title)
        
        previews_container = QWidget()
        previews_layout = QHBoxLayout()
        previews_container.setLayout(previews_layout)
        
        # Axial preview
        axial_container = QWidget()
        axial_layout = QVBoxLayout()
        axial_container.setLayout(axial_layout)
        axial_label = QLabel("AXIAL")
        axial_label.setAlignment(Qt.AlignCenter)
        axial_label.setStyleSheet("font-weight: bold; color: red; margin-bottom: 5px;")
        axial_layout.addWidget(axial_label)
        self.axial_preview = QVTKRenderWindowInteractor(axial_container)
        self.axial_preview.setMinimumSize(180, 150)
        self.axial_preview.setMaximumSize(180, 150)
        axial_layout.addWidget(self.axial_preview)
        previews_layout.addWidget(axial_container)
        
        # Coronal preview
        coronal_container = QWidget()
        coronal_layout = QVBoxLayout()
        coronal_container.setLayout(coronal_layout)
        coronal_label = QLabel("CORONAL")
        coronal_label.setAlignment(Qt.AlignCenter)
        coronal_label.setStyleSheet("font-weight: bold; color: green; margin-bottom: 5px;")
        coronal_layout.addWidget(coronal_label)
        self.coronal_preview = QVTKRenderWindowInteractor(coronal_container)
        self.coronal_preview.setMinimumSize(180, 150)
        self.coronal_preview.setMaximumSize(180, 150)
        coronal_layout.addWidget(self.coronal_preview)
        previews_layout.addWidget(coronal_container)
        
        # Sagittal preview
        sagittal_container = QWidget()
        sagittal_layout = QVBoxLayout()
        sagittal_container.setLayout(sagittal_layout)
        sagittal_label = QLabel("SAGITTAL")
        sagittal_label.setAlignment(Qt.AlignCenter)
        sagittal_label.setStyleSheet("font-weight: bold; color: blue; margin-bottom: 5px;")
        sagittal_layout.addWidget(sagittal_label)
        self.sagittal_preview = QVTKRenderWindowInteractor(sagittal_container)
        self.sagittal_preview.setMinimumSize(180, 150)
        self.sagittal_preview.setMaximumSize(180, 150)
        sagittal_layout.addWidget(self.sagittal_preview)
        previews_layout.addWidget(sagittal_container)
        
        previews_layout.addStretch()
        bottom_layout.addWidget(previews_container)
        parent_layout.addWidget(bottom_panel)
    
    def set_main_view(self):
        self.center_stack.setCurrentIndex(0)
        self.vtk_app.is_slice_view_active = False
        self.main_view_btn.setStyleSheet('background-color:#4CAF50; color:white')
        self.slice_view_btn.setStyleSheet('')
        self.status_bar.showMessage("Main View - 3D Volume Rendering")
    
    def set_slice_view(self):
        self.center_stack.setCurrentIndex(1)
        self.vtk_app.is_slice_view_active = True
        
        # Initialize slice view widgets if needed
        for widget in [self.axial_slice_view, self.coronal_slice_view, self.sagittal_slice_view]:
            ren_win = widget.GetRenderWindow()
            if ren_win.GetRenderers().GetNumberOfItems() == 0:
                ren = vtk.vtkRenderer()
                ren.SetBackground(0.1, 0.1, 0.1)
                ren_win.AddRenderer(ren)
                inter = ren_win.GetInteractor()
                style = vtk.vtkInteractorStyleImage()
                inter.SetInteractorStyle(style)
                inter.Initialize()
        
        # Update slice views
        self.vtk_app.update_slice_view_widgets()
        
        self.main_view_btn.setStyleSheet('')
        self.slice_view_btn.setStyleSheet('background-color:#4CAF50; color:white')
        self.status_bar.showMessage("Slice View - Multi-planar Reconstruction")
    
    def load_dicom_model(self):
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select DICOM Directory",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            self.load_model_from_directory(directory)
    
    def load_model_from_directory(self, directory_path):
        if not os.path.exists(directory_path):
            QMessageBox.warning(self, "Error", f"Directory not found: {directory_path}")
            return
        
        dicom_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_lower = file.lower()
                file_path = os.path.join(root, file)

                if file_lower.endswith('.dcm'):
                    dicom_files.append(file_path)
                elif file_lower.endswith(('.dicom', '.ima')):
                    dicom_files.append(file_path)
                elif '.' not in file and 1024 <= os.path.getsize(file_path) <= 50 * 1024 * 1024:
                    try:
                        with open(file_path, 'rb') as f:
                            header = f.read(132)
                            if len(header) >= 132 and header[128:132] == b'DICM':
                                dicom_files.append(file_path)
                    except:
                        pass

        print(f"FOUND {len(dicom_files)} DICOM files (including subfolders)")
        
        if not dicom_files:
            QMessageBox.warning(self, "No DICOM Files", 
                            f"No DICOM files found in: {directory_path}\n\n"
                            f"Supported formats: .dcm, .dicom, .ima files")
            return
        
        print(f"SUCCESS: Found {len(dicom_files)} DICOM files")
        
        model_name = os.path.basename(directory_path)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f"Loading {len(dicom_files)} DICOM files from: {model_name}...")
        
        self.load_thread = LoadModelThread(dicom_files, directory_path)
        self.load_thread.progress.connect(self.progress_bar.setValue)
        self.load_thread.finished.connect(self.on_worker_ready_to_load)
        self.load_thread.error.connect(self.on_model_load_error)
        self.load_thread.start()
    
    def on_model_loaded(self, message):
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(message)
        
        model_name = os.path.basename(self.load_thread.directory_path)
        self.loaded_models[model_name] = self.load_thread.directory_path
        
        item = QListWidgetItem(f"📁 {model_name}")
        self.models_list.addItem(item)
        
        self.models_list.setCurrentItem(item)
        self.on_model_selected(item)
        
        # Initialize previews
        self.setup_slice_previews()
        
        # Initialize slice sliders
        self.axial_slider.setValue(500)
        self.coronal_slider.setValue(500)
        self.sagittal_slider.setValue(500)
        
        if self.vtk_app.current_dataset:
            QMessageBox.information(self, "Success", f"Model '{model_name}' loaded successfully!")
        else:
            QMessageBox.warning(self, "Warning", f"Model '{model_name}' loaded but visualization may not be working correctly.")
    
    def on_model_load_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Model loading failed")
        QMessageBox.critical(self, "Loading Error", error_message)
    
    def on_worker_ready_to_load(self, message, dicom_files, directory_path):
        # This runs on the main (GUI) thread — safe to call VTK
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Loading dataset into VTK...")
        success = self.vtk_app.load_dicom_dataset(dicom_files)
        model_name = os.path.basename(directory_path)
        if success:
            # Register model (use a unique key, see patch below)
            self.loaded_models[model_name] = directory_path
            item = QListWidgetItem(f"📁 {model_name}")
            self.models_list.addItem(item)
            self.models_list.setCurrentItem(item)
            self.on_model_selected(item)
            self.setup_slice_previews()
            self.axial_slider.setValue(500)
            self.coronal_slider.setValue(500)
            self.sagittal_slider.setValue(500)
            QMessageBox.information(self, "Success", f"Model '{model_name}' loaded successfully!")
        else:
            QMessageBox.warning(self, "Warning", f"Model '{model_name}' loaded but visualization may not be working correctly.")

    def on_model_selected(self, item):
        if item:
            model_name = item.text().replace("📁 ", "")
            if model_name in self.loaded_models:
                self.current_model = model_name
                self.current_model_label.setText(f"Current: {model_name}")
                
                # Switch dataset in VTK
                dataset_key = os.path.abspath(self.loaded_models[model_name])
                if dataset_key in self.vtk_app.loaded_datasets:
                    self.vtk_app.current_dataset = dataset_key
                    self.vtk_app.reader = self.vtk_app.loaded_datasets[dataset_key]['reader']
                    self.vtk_app.update_visualization()
                
                self.status_bar.showMessage(f"Switched to model: {model_name}")
                
                # Update slice previews
                self.setup_slice_previews()
    
    def remove_model(self):
        current_item = self.models_list.currentItem()
        if not current_item:
            return

        model_name = current_item.text().replace("📁 ", "")

        reply = QMessageBox.question(
            self,
            "Remove Model",
            f"Are you sure you want to remove '{model_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        dataset_key = os.path.abspath(self.loaded_models.get(model_name, ""))  # compute before deleting if needed
        if dataset_key in self.vtk_app.loaded_datasets:
            # remove VTK reader entry and attempt to free it
            try:
                del self.vtk_app.loaded_datasets[dataset_key]
            except:
                pass

        # Remove from list widget
        self.models_list.takeItem(self.models_list.row(current_item))

        # If user removed the active model → FULL RESET
        if self.current_model == model_name:
            self.current_model = None
            self.current_model_label.setText("No model loaded")

            # VTK full reset
            self.vtk_app.reset_before_load()

            # Reset preview widgets
            for widget in [
                self.axial_preview,
                self.coronal_preview,
                self.sagittal_preview,
                self.axial_slice_view,
                self.coronal_slice_view,
                self.sagittal_slice_view,
            ]:
                ren = widget.GetRenderWindow().GetRenderers().GetFirstRenderer()
                if ren:
                    ren.RemoveAllViewProps()
                    widget.GetRenderWindow().Render()

        self.status_bar.showMessage(f"Removed model: {model_name}")

    def show_model_info(self):
        if self.current_model and self.current_model in self.loaded_models:
            model_info = f"""
Model Name: {self.current_model}
Directory: {self.loaded_models[self.current_model]}

To view detailed DICOM information, load the model first.
"""
            QMessageBox.information(self, "Model Information", model_info)
        else:
            QMessageBox.information(self, "Info", "No model selected or loaded.")
    
    def reset_view(self):
        self.vtk_app.reset_view()
    
    def toggle_measurement(self):
        if not self.is_measuring:
            self.vtk_app.start_measurement()
            self.is_measuring = True
            self.measure_btn.setText("Stop Measure")
            self.measure_btn.setStyleSheet("""
                QPushButton {
                    background-color: #F44336;
                    color: white;
                    font-weight: bold;
                    padding: 6px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background-color: #D32F2F;
                }
            """)
            self.status_bar.showMessage("MEASUREMENT MODE: Click on two points on the anatomy to measure distance")
        else:
            self.vtk_app.stop_measurement()
            self.is_measuring = False
            self.measure_btn.setText("Start Measure")
            self.measure_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    font-weight: bold;
                    padding: 6px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
            """)
            self.status_bar.showMessage("Measurement mode stopped")
    
    def clear_measurements(self):
        self.vtk_app.clear_measurements()
        self.status_bar.showMessage("All measurements cleared")
    
    def show_help(self):
        help_text = """
Medical DICOM Viewer - Help

View Modes:
- Main View: 3D volume rendering with coordinate system
- Slice View: Large multi-planar reconstruction views

Controls:
- Slice Sliders: Control axial, coronal, and sagittal slices
- Rotation Buttons: Rotate 3D view around X, Y, Z axes
- Camera Views: Predefined front, side, top views
- Measurement: Click two points to measure distance

Navigation:
- Left-click + drag: Rotate 3D view
- Right-click + drag: Zoom
- Middle-click + drag: Pan

Measurement:
1. Click "Start Measure"
2. Click on two points in the anatomy
3. Distance displays in mm and pixels
4. Click "Clear" to remove measurements
"""
        QMessageBox.information(self, "Help", help_text)
    
    def setup_slice_previews(self):
        if not self.vtk_app.reader:
            print("No DICOM reader for slice previews")
            return
        
        try:
            for widget in [self.axial_preview, self.coronal_preview, self.sagittal_preview]:
                ren_win = widget.GetRenderWindow()
                if ren_win:
                    ren_win.GetRenderers().RemoveAllItems()
                    ren = vtk.vtkRenderer()
                    ren.SetBackground(0.1, 0.1, 0.1)
                    ren_win.AddRenderer(ren)
                    ren_win.SetSize(180, 150)
            
            self.vtk_app.update_preview_widgets()
            print("Slice previews initialized")
            
        except Exception as e:
            print(f"Error setting up slice previews: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_slice_previews(self):
        for widget in [self.axial_preview, self.coronal_preview, self.sagittal_preview]:
            if widget:
                ren_win = widget.GetRenderWindow()
                if ren_win:
                    ren_win.GetRenderers().RemoveAllItems()
                    widget.GetRenderWindow().Render()
                    
    def update_window_level(self):
        """Update window/level based on slider values"""
        window_value = self.window_slider.value()
        level_value = self.level_slider.value()
        
        self.window_value_label.setText(str(window_value))
        self.level_value_label.setText(str(level_value))
        
        self.vtk_app.set_window_level(window_value, level_value)

    def update_opacity(self):
        """Update inner/outer opacity"""
        inner_opacity = self.inner_opacity_slider.value() / 100.0
        outer_opacity = self.outer_opacity_slider.value() / 100.0
        
        self.inner_value_label.setText(f"{int(inner_opacity*100)}%")
        self.outer_value_label.setText(f"{int(outer_opacity*100)}%")
        
        self.vtk_app.adjust_opacity_range(inner_opacity, outer_opacity)
            
    def closeEvent(self, event):
        """Clean up VTK resources before closing"""
        try:
            # Clean up VTK widgets
            widgets = [
                self.vtk_widget,
                self.axial_preview, 
                self.coronal_preview,
                self.sagittal_preview,
                self.axial_slice_view,
                self.coronal_slice_view,
                self.sagittal_slice_view
            ]
            
            for widget in widgets:
                if widget:
                    try:
                        ren_win = widget.GetRenderWindow()
                        if ren_win:
                            # Remove all renderers
                            renderers = ren_win.GetRenderers()
                            if renderers:
                                renderers.RemoveAllItems()
                            # Finalize the window
                            ren_win.Finalize()
                    except:
                        pass
                    
            # Clean up VTK app
            if hasattr(self, 'vtk_app'):
                self.vtk_app.cleanup()
                    
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        event.accept()

    def set_preset(self, preset_type):
        """Set preset window/level values"""
        presets = {
            'soft_tissue': (400, 40),
            'bone': (2000, 300),
            'lung': (1500, -500)
        }
        
        if preset_type in presets:
            window, level = presets[preset_type]
            self.window_slider.setValue(window)
            self.level_slider.setValue(level)
            self.update_window_level()
            
    def save_model_state(self):
        """Save the current model state to a file"""
        
        if not self.vtk_app.current_dataset:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
        
        # Get save file path
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model State",
            "",
            "Model State Files (*.mstate);;All Files (*)"
        )
        
        if filepath:
            # Ensure .mstate extension
            if not filepath.endswith('.mstate'):
                filepath += '.mstate'
            
            # Get current DICOM directory
            dicom_dir = self.loaded_models.get(self.current_model, "")
            
            # Save additional metadata
            metadata = {
                'model_name': self.current_model,
                'dicom_directory': dicom_dir,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Combine metadata with VTK state
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'metadata': metadata,
                        'vtk_state': {
                            'window_value': self.vtk_app.window_value,
                            'level_value': self.vtk_app.level_value,
                            'measurement_points': self.vtk_app.measurement_points
                        }
                    }, f)
                
                QMessageBox.information(self, "Success", 
                                    f"Model state saved to:\n{filepath}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                f"Failed to save model: {str(e)}")

    def load_model_state(self):
        """Load a previously saved model state"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model State",
            "",
            "Model State Files (*.mstate);;All Files (*)"
        )
        
        if filepath:
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                metadata = data.get('metadata', {})
                vtk_state = data.get('vtk_state', {})
                
                # Check if DICOM directory exists
                dicom_dir = metadata.get('dicom_directory', '')
                if not os.path.exists(dicom_dir):
                    # Ask user to locate DICOM directory
                    dicom_dir = QFileDialog.getExistingDirectory(
                        self,
                        "Locate DICOM Directory",
                        "",
                        QFileDialog.ShowDirsOnly
                    )
                
                if dicom_dir and os.path.exists(dicom_dir):
                    # Load the model first
                    self.load_model_from_directory(dicom_dir)
                    
                    # Restore VTK state
                    self.vtk_app.window_value = vtk_state.get('window_value', 400)
                    self.vtk_app.level_value = vtk_state.get('level_value', 40)
                    self.vtk_app.measurement_points = vtk_state.get('measurement_points', [])
                    
                    # Update UI
                    self.window_slider.setValue(self.vtk_app.window_value)
                    self.level_slider.setValue(self.vtk_app.level_value)
                    self.update_window_level()
                    
                    # Recreate measurements
                    self.vtk_app.clear_measurements()
                    for i in range(0, len(self.vtk_app.measurement_points), 2):
                        if i + 1 < len(self.vtk_app.measurement_points):
                            self.vtk_app.create_measurement_point_marker(
                                self.vtk_app.measurement_points[i]
                            )
                            self.vtk_app.create_measurement_point_marker(
                                self.vtk_app.measurement_points[i + 1]
                            )
                    
                    QMessageBox.information(self, "Success", 
                                        f"Model state loaded:\n{metadata.get('model_name', 'Unknown')}")
                else:
                    QMessageBox.warning(self, "Error", 
                                    "Could not find DICOM directory.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                f"Failed to load model: {str(e)}")

    def open_color_dialog(self):
        if not self.vtk_app or not self.vtk_app.volume:
            QMessageBox.warning(self, "Error", "Load a model first.")
            return

        color = QColorDialog.getColor()

        if color.isValid():
            r = color.red() / 255.0
            g = color.green() / 255.0
            b = color.blue() / 255.0

            # Update preview button color
            self.color_button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            )

            # Update VTK color transfer function
            self.vtk_app.update_volume_color((r, g, b))

    def export_view_image(self):
        """Export current view as an image"""
        if not self.vtk_app.current_dataset:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
        
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export View as Image",
            f"{self.current_model}_view",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)"
        )
        
        if filepath:
            # Determine format from filter
            if 'PNG' in selected_filter:
                format = 'PNG'
                if not filepath.endswith('.png'):
                    filepath += '.png'
            else:
                format = 'JPEG'
                if not filepath.endswith(('.jpg', '.jpeg')):
                    filepath += '.jpg'
            
            success = self.vtk_app.export_model_as_image(filepath, format)
            if success:
                QMessageBox.information(self, "Success", 
                                    f"Image exported to:\n{filepath}")
            else:
                QMessageBox.warning(self, "Error", 
                                "Failed to export image.")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    print("DEBUG: App started")

    # Create main window, but keep it hidden for now
    window = MainWindow()
    window.hide()

    # Create & show the animated splash screen
    splash = AnimatedSplashScreen("Advanced DICOM Medical Visualizer.gif")
    splash.show()
    print("DEBUG: Splash shown")

    # After a few seconds, close splash and show main window
    def show_main_window():
        print("DEBUG: Timer triggered, closing splash and showing main window")
        splash.close()
        window.showFullScreen()   # or window.show()

    # Show main window after 3 seconds (3000 ms)
    QTimer.singleShot(3000, show_main_window)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
