// import * as Cesium from 'cesium';

// Your access token can be found at: https://ion.cesium.com/tokens.
// This is the default access token from your ion account

Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI0YTZhOTAwYy0wMmE2LTQwYWQtOGViZS0xMDM3MDE1ZWQzODkiLCJpZCI6MTU0MzcwLCJpYXQiOjE2ODk1NjQxMTd9.AJHX5iqrqsxONF7D70YuQc5CNYPkG5m4ltTZ4cH8QTg';

// Initialize the Cesium Viewer in the HTML element with the `cesiumContainer` ID.
const viewer = new Cesium.Viewer("cesiumContainer", {
    baseLayer: Cesium.ImageryLayer.fromProviderAsync(
        Cesium.TileMapServiceImageryProvider.fromUrl(
            Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII")
        )
    ),
    baseLayerPicker: false,
    geocoder: false,
});

// Add Cesium OSM Buildings, a global 3D buildings layer.
// const buildingTileset = await Cesium.createOsmBuildingsAsync();
// viewer.scene.primitives.add(buildingTileset); 

// load 3d tileset data
const tileset = await Cesium.Cesium3DTileset.fromUrl("./root.json");
viewer.scene.primitives.add(tileset);

// fly to singapore
viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(103.851959, 1.290270, 1000.0),
    orientation: {
        heading: Cesium.Math.toRadians(0.0),
        pitch: Cesium.Math.toRadians(-90.0),
    },
});
