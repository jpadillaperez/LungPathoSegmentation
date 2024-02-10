# Author: Matthias Keicher, CAMP @ Technical University of Munich
# E-Mail: matthias.keicher@tum.de
# vim: set ts=4 sw=4

import SimpleITK as sitk


itr = 0
iteration = 1


def iteration_callback(filter):
    global itr
    #print("deformable iter:", itr, "loss:", filter.GetMetricValue(), flush=True)
    itr += 1


def deformable_registration(input_fixed, input_moving, spacing):
    #### See: https://simpleitk.org/SPIE2019_COURSE/05_advanced_registration.html
    input_fixed_sitk = sitk.GetImageFromArray(input_fixed.astype('float'), False)
    input_fixed_sitk.SetSpacing(spacing)
    input_moving_sitk = sitk.GetImageFromArray(input_moving.astype('float'), False) # Nifti file instead of numpy
    input_moving_sitk.SetSpacing(spacing)

    registration_method = sitk.ImageRegistrationMethod()

    grid_physical_spacing = [50.0, 50.0, 50.0]  # A control point every 50mm
    image_physical_size = [size * spacing for size, spacing in zip(input_fixed_sitk.GetSize(), input_fixed_sitk.GetSpacing())]

    mesh_size = [int(image_size / grid_spacing + 0.5) for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
    mesh_size = [int(sz / 4 + 0.5) for sz in mesh_size]

    initial_transform = sitk.BSplineTransformInitializer(image1=input_fixed_sitk, transformDomainMeshSize=mesh_size, order=3)


    registration_method.SetInitialTransformAsBSpline(initial_transform, inPlace=False, scaleFactors=[1, 2, 4])

    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    # Differenrent opitimizer

    #registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-2, numberOfIterations=100,
    #                                         deltaConvergenceTolerance=0.01)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()


    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(registration_method))

    final_transformation = registration_method.Execute(input_fixed_sitk, input_moving_sitk)

    return final_transformation



def transform_moving_volume(input_fixed, input_moving, final_transformation, spacing, method=sitk.sitkNearestNeighbor):
    input_fixed_sitk = sitk.GetImageFromArray(input_fixed.astype('float'), False)
    input_fixed_sitk.SetSpacing(spacing)
    input_moving_sitk = sitk.GetImageFromArray(input_moving.astype('float'), False)  # Nifti file instead of numpy
    input_moving_sitk.SetSpacing(spacing)

    transformed = sitk.Resample(input_moving_sitk, input_fixed_sitk, final_transformation,
                                             method, 0.0, input_moving_sitk.GetPixelID())

    return sitk.GetArrayFromImage(transformed)




