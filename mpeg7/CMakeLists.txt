
SET(SRC_FILES
	Convert.cpp
	Feature.cpp	
	Frame.cpp
	
	AddressLib/address.c	
	AddressLib/inter.c
	AddressLib/intra.c
	AddressLib/nhood.c
	AddressLib/vopio.c
	
	Descriptors/ColorLayout.cpp
	Descriptors/ColorQuant.cpp
	Descriptors/ColorSpace.cpp
	Descriptors/ColorStructure.cpp
	Descriptors/ContourShape.cpp
	Descriptors/DominantColor.cpp
	Descriptors/EdgeHist.cpp
	Descriptors/FaceRecognition.cpp
	Descriptors/HomoTexture.cpp
	Descriptors/RegionShape.cpp
	Descriptors/ScalableColor.cpp
	
	Extraction/ColorLayoutExtraction.cpp
	Extraction/ColorSpaceExtraction.cpp
	Extraction/ColorStructureExtraction.cpp
	Extraction/ContourShapeExtraction.cpp
	Extraction/DominantColorExtraction.cpp
	Extraction/EdgeHistExtraction.cpp
	Extraction/FaceRecognitionExtraction.cpp
	Extraction/GoFGoPColorExtraction.cpp
	Extraction/HomoTextureExtraction.cpp
	Extraction/homo_texture_extractor.cpp
	Extraction/imgutil.c
	Extraction/RegionShapeExtraction.cpp
	Extraction/ScalableColorExtraction.cpp
)

SET(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/AddressLib
	${CMAKE_CURRENT_SOURCE_DIR}/Descriptors
	${CMAKE_CURRENT_SOURCE_DIR}/Extraction
    ${OpenCV_INCLUDE_DIRS}
)

SET(LIBRARIES
	${OpenCV_LIBS}
)

# Make sure the compiler can find include files from our library.
INCLUDE_DIRECTORIES(${INCLUDE_DIRS})

# Add binary that is built from the source file "main.cpp".
# The extension is automatically found.
ADD_LIBRARY(mpeg7 ${SRC_FILES})
TARGET_LINK_LIBRARIES(mpeg7 ${LIBRARIES})

INSTALL(TARGETS mpeg7
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT runtime
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT devel
        ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT devel)
		
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
		DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/AddressLib/
		DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Descriptors/
		DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Extraction/
		DESTINATION "${INSTALL_INCLUDE_DIR}"
		COMPONENT devel 
		FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" 
		PATTERN ".svn" EXCLUDE)

