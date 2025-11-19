import slicer
import numpy as np
import vtk

# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------
known_length_mm = 10.0  # physical length of scale bar
fiducial_node_name = "F"  # name of your markups node
# ---------------------------------------------------------

# 1. GET THE RAW VOLUME
fid = slicer.util.getNode(fiducial_node_name)
if fid.GetNumberOfControlPoints() < 2:
    raise RuntimeError("Fiducial node must have at least 2 points.")

lm = slicer.app.layoutManager()
red_widget = lm.sliceWidget("Red")
red_logic = red_widget.sliceLogic()
bg_vol_id = red_logic.GetSliceCompositeNode().GetBackgroundVolumeID()
working_vol = slicer.mrmlScene.GetNodeByID(bg_vol_id)

if not working_vol:
    raise RuntimeError("No background volume found in Red view.")

print(f"Processing volume: {working_vol.GetName()}")

# 2. CALIBRATE SPACING (Directly on the RGB image)
p1_ras = np.array(fid.GetNthControlPointPosition(0))
p2_ras = np.array(fid.GetNthControlPointPosition(1))

# Use the working volume's matrix
ras_to_ijk = vtk.vtkMatrix4x4()
working_vol.GetRASToIJKMatrix(ras_to_ijk)


def ras_to_ijk_point(ras):
    ras4 = [ras[0], ras[1], ras[2], 1.0]
    ijk4 = [0.0, 0.0, 0.0, 0.0]
    ras_to_ijk.MultiplyPoint(ras4, ijk4)
    return np.array(ijk4[:3])


p1_ijk = ras_to_ijk_point(p1_ras)
p2_ijk = ras_to_ijk_point(p2_ras)

delta_ijk = p2_ijk - p1_ijk
pixel_distance = np.linalg.norm(delta_ijk[:2])

if pixel_distance == 0:
    raise RuntimeError("Fiducial points are at the same position.")

pixel_spacing_mm = known_length_mm / pixel_distance
print(f"Computed spacing: {pixel_spacing_mm:.4f} mm/pixel")

# Update spacing of the RGB volume directly
spacing = list(working_vol.GetSpacing())
spacing[0] = pixel_spacing_mm
spacing[1] = pixel_spacing_mm
working_vol.SetSpacing(spacing)
print("Image spacing calibrated.")

# 3. SETUP HIGH-RES SEGMENTATION
# Create a new segmentation node
seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "seg")
seg_node.CreateDefaultDisplayNodes()

# *** CRITICAL STEP *** # Force segmentation geometry to match the RGB volume exactly.
# This ensures the "grid" you paint on has the exact same number of pixels as your JPEG.
seg_node.SetReferenceImageGeometryParameterFromVolumeNode(working_vol)
print("Segmentation geometry locked to Reference Volume.")

# 4. UPDATE VIEW
# Switch to Segment Editor automatically
slicer.util.selectModule("SegmentEditor")

# Set the Master Volume in Segment Editor to our RGB volume
plugin_handler = slicer.qSlicerSegmentEditorAbstractEffect.pluginHandler()
if plugin_handler:
    editor_widget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
    editor_widget.setSegmentationNode(seg_node)
    editor_widget.setMasterVolumeNode(working_vol)

    # Activate the Paint tool automatically
    editor_widget.setActiveEffectByName("Paint")

print("Setup complete! You can paint on the Color image now.")
