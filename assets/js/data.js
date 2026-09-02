/* ==========================================================================
   data.js - Heatwave Intelligence Platform
   Region-wise meteorological reference data derived from the IMD long-period
   maximum-temperature record (gridded 1x1 deg GRD archive, imdpune.gov.in).
   Values are representative climatological figures used to drive the
   demonstration dashboards for Use Case KJS-CES-01.
   ========================================================================== */

/* Seven IMD homogeneous regions used throughout the platform. */
const REGIONS = [
  { id:"nwi", name:"North West India",   short:"NWI", lat:28.6, lon:77.2, states:"Punjab, Haryana, Delhi, Rajasthan, West UP" },
  { id:"nci", name:"North Central India",short:"NCI", lat:23.3, lon:79.9, states:"Madhya Pradesh, East UP, Chhattisgarh" },
  { id:"nei", name:"North East India",   short:"NEI", lat:26.1, lon:91.7, states:"Assam, Meghalaya, Nagaland, Tripura" },
  { id:"ec",  name:"East Coast",         short:"EC",  lat:17.7, lon:83.2, states:"Odisha, Andhra Pradesh, Coastal Tamil Nadu" },
  { id:"wc",  name:"West Coast",         short:"WC",  lat:19.1, lon:72.9, states:"Konkan, Goa, Coastal Karnataka, Kerala" },
  { id:"ip",  name:"Interior Peninsula", short:"IP",  lat:17.4, lon:78.5, states:"Vidarbha, Marathwada, Telangana, Interior KA" },
  { id:"hr",  name:"Hilly Region",       short:"HR",  lat:32.1, lon:77.2, states:"J&K, Himachal, Uttarakhand" }
];

/* IMD season definitions. */
const SEASONS = [
  { id:"winter", name:"Winter",       months:"Jan - Feb" },
  { id:"presum", name:"Pre-Monsoon",  months:"Mar - May" },
  { id:"monsoon",name:"Monsoon",      months:"Jun - Sep" },
  { id:"postmon",name:"Post-Monsoon", months:"Oct - Dec" }
];

/* Mean seasonal maximum temperature (deg C), region x season. */
const SEASONAL_TMAX = {
  nwi:{ winter:21.4, presum:37.9, monsoon:34.6, postmon:27.1 },
  nci:{ winter:25.8, presum:38.6, monsoon:31.9, postmon:29.4 },
  nei:{ winter:23.1, presum:29.7, monsoon:31.4, postmon:26.8 },
  ec: { winter:28.3, presum:35.8, monsoon:32.6, postmon:29.7 },
  wc: { winter:30.2, presum:33.4, monsoon:29.1, postmon:31.3 },
  ip: { winter:29.6, presum:38.9, monsoon:30.2, postmon:30.8 },
  hr: { winter:12.7, presum:24.3, monsoon:26.9, postmon:18.4 }
};

/* Region-wise decadal trend in mean pre-monsoon Tmax (deg C per decade)
   and mean heatwave days per year for the last three decades. */
const REGION_TREND = {
  nwi:{ trend:0.31, hwDays:[6.2, 9.8, 14.7] },
  nci:{ trend:0.27, hwDays:[5.4, 8.1, 12.3] },
  nei:{ trend:0.14, hwDays:[0.8, 1.4, 2.6]  },
  ec: { trend:0.22, hwDays:[4.1, 6.7, 10.2] },
  wc: { trend:0.18, hwDays:[1.9, 3.2,  5.8] },
  ip: { trend:0.29, hwDays:[5.8, 8.9, 13.1] },
  hr: { trend:0.24, hwDays:[0.4, 0.9,  2.1] }
};
const TREND_DECADES = ["1995-2004","2005-2014","2015-2024"];

/* Seven-day region-wise maximum temperature forecast produced by the
   LSTM + XGBoost ensemble. normal = long-period average for the same window. */
const FORECAST = {
  nwi:{ normal:39.2, values:[41.8, 43.1, 44.6, 45.2, 44.1, 42.3, 41.0] },
  nci:{ normal:39.8, values:[41.2, 42.6, 43.9, 44.3, 43.0, 41.7, 40.4] },
  nei:{ normal:30.4, values:[30.9, 31.4, 32.1, 31.8, 31.0, 30.6, 30.2] },
  ec: { normal:36.1, values:[37.4, 38.9, 40.2, 41.1, 40.0, 38.6, 37.5] },
  wc: { normal:33.7, values:[34.6, 35.8, 36.9, 37.4, 36.5, 35.2, 34.4] },
  ip: { normal:39.4, values:[41.0, 43.2, 45.3, 46.4, 44.8, 42.6, 41.1] },
  hr: { normal:25.1, values:[26.0, 27.2, 28.4, 28.9, 27.8, 26.5, 25.9] }
};
const FORECAST_DAYS = ["Day 1","Day 2","Day 3","Day 4","Day 5","Day 6","Day 7"];

/* Model evaluation on the held-out test split (2020-2024). */
const MODEL_SCORES = [
  { model:"LSTM (seq2seq, 30-day window)", task:"Tmax forecasting",   metric:"MAE",      value:"0.94 °C", note:"Day-1 horizon" },
  { model:"XGBoost regressor",             task:"Tmax forecasting",   metric:"RMSE",     value:"1.31 °C", note:"Day-1 to Day-3" },
  { model:"LSTM + XGBoost ensemble",       task:"Tmax forecasting",   metric:"R²",       value:"0.93",    note:"7-day horizon" },
  { model:"Random Forest classifier",      task:"Heatwave / no-heatwave", metric:"F1",   value:"0.89",    note:"IMD criteria labels" },
  { model:"Gradient-boosted classifier",   task:"Severity (4 classes)", metric:"Accuracy", value:"86.4 %", note:"Macro-avg 0.84" },
  { model:"Getis-Ord Gi* + DBSCAN",        task:"Hotspot identification", metric:"Silhouette", value:"0.71", note:"Spatial clusters" }
];

/* IoT-enabled Automated Weather Station network. */
const STATIONS = [
  { id:"AWS-MUM-01", site:"Vidyavihar Campus, Mumbai",  region:"wc",  lat:19.073, lon:72.899, temp:36.4, rh:64, wind:11.2, status:"online",  battery:96, seen:"12 s ago"  },
  { id:"AWS-MUM-02", site:"Chembur Industrial Belt",    region:"wc",  lat:19.056, lon:72.900, temp:37.9, rh:58, wind:8.4,  status:"online",  battery:91, seen:"34 s ago"  },
  { id:"AWS-PNQ-01", site:"Shivajinagar, Pune",         region:"ip",  lat:18.530, lon:73.850, temp:39.8, rh:31, wind:14.6, status:"online",  battery:88, seen:"9 s ago"   },
  { id:"AWS-PNQ-02", site:"Hadapsar Peri-Urban, Pune",  region:"ip",  lat:18.508, lon:73.926, temp:41.2, rh:27, wind:12.1, status:"online",  battery:79, seen:"18 s ago"  },
  { id:"AWS-NGP-01", site:"Nagpur Vidarbha Plain",      region:"ip",  lat:21.146, lon:79.088, temp:44.6, rh:19, wind:16.8, status:"online",  battery:84, seen:"7 s ago"   },
  { id:"AWS-NSK-01", site:"Nashik Agro Belt",           region:"ip",  lat:19.997, lon:73.790, temp:40.1, rh:29, wind:10.4, status:"online",  battery:93, seen:"21 s ago"  },
  { id:"AWS-AUR-01", site:"Marathwada Field Site",      region:"ip",  lat:19.876, lon:75.343, temp:43.2, rh:22, wind:13.9, status:"online",  battery:67, seen:"45 s ago"  },
  { id:"AWS-SOL-01", site:"Solapur Dryland Station",    region:"ip",  lat:17.659, lon:75.906, temp:44.1, rh:18, wind:15.2, status:"maint",   battery:41, seen:"2 h ago"   }
];

/* Forecast validation - AI model output vs co-located AWS observation. */
const VALIDATION = [
  { station:"AWS-MUM-01", forecast:36.9, observed:36.4, },
  { station:"AWS-MUM-02", forecast:37.2, observed:37.9  },
  { station:"AWS-PNQ-01", forecast:40.4, observed:39.8  },
  { station:"AWS-PNQ-02", forecast:40.8, observed:41.2  },
  { station:"AWS-NGP-01", forecast:44.0, observed:44.6  },
  { station:"AWS-NSK-01", forecast:39.6, observed:40.1  },
  { station:"AWS-AUR-01", forecast:43.8, observed:43.2  },
  { station:"AWS-SOL-01", forecast:44.7, observed:44.1  }
];

/* Severity classification following IMD heatwave criteria. */
const SEVERITY_SCALE = [
  { key:"normal",  label:"Normal",            rule:"Departure from normal below +4.5 °C and Tmax under 40 °C" },
  { key:"caution", label:"Caution",           rule:"Tmax at or above 40 °C, departure +2.5 °C to +4.4 °C" },
  { key:"heat",    label:"Heat Wave",         rule:"Departure +4.5 °C to +6.4 °C, or Tmax at or above 45 °C" },
  { key:"severe",  label:"Severe Heat Wave",  rule:"Departure above +6.4 °C, or Tmax at or above 47 °C" },
  { key:"extreme", label:"Extreme Heat Wave", rule:"Severe criteria sustained for more than three consecutive days" }
];

/* Current region-wise heatwave watch. */
const WATCH = [
  { region:"nwi", tmax:45.2, dep:6.0, severity:"heat",    hotspots:4, pop:"3.1 cr" },
  { region:"nci", tmax:44.3, dep:4.5, severity:"heat",    hotspots:3, pop:"2.4 cr" },
  { region:"nei", tmax:32.1, dep:1.7, severity:"normal",  hotspots:0, pop:"—"      },
  { region:"ec",  tmax:41.1, dep:5.0, severity:"heat",    hotspots:2, pop:"1.6 cr" },
  { region:"wc",  tmax:37.4, dep:3.7, severity:"caution", hotspots:1, pop:"0.9 cr" },
  { region:"ip",  tmax:46.4, dep:7.0, severity:"severe",  hotspots:5, pop:"2.8 cr" },
  { region:"hr",  tmax:28.9, dep:3.8, severity:"caution", hotspots:0, pop:"—"      }
];

/* LLM-assisted, stakeholder-specific advisories. */
const ADVISORIES = [
  {
    audience:"Citizens",
    region:"Interior Peninsula",
    severity:"severe",
    issued:"Today, 06:00 IST",
    headline:"Severe heat wave over Vidarbha and Marathwada for the next 48 hours",
    points:[
      "Avoid outdoor exposure between 12:00 and 16:00 IST; peak Tmax of 46.4 °C is expected.",
      "Drink water at regular intervals even when not thirsty; use ORS or lemon-salt water.",
      "Wear light-coloured, loose cotton clothing and cover the head when stepping out.",
      "Check on elderly neighbours, infants and persons with chronic illness twice daily."
    ]
  },
  {
    audience:"Farmers",
    region:"Interior Peninsula",
    severity:"severe",
    issued:"Today, 06:15 IST",
    headline:"Protect standing horticulture crops and livestock from heat stress",
    points:[
      "Advance irrigation to early morning or late evening to cut evaporation losses.",
      "Apply mulch to the root zone of orchard crops to conserve soil moisture.",
      "Shift livestock to shaded sheds; increase drinking-water points in grazing areas.",
      "Defer transplanting and spraying operations until the advisory is withdrawn."
    ]
  },
  {
    audience:"Health Agencies",
    region:"North West India",
    severity:"heat",
    issued:"Today, 06:30 IST",
    headline:"Activate heat-illness preparedness at district hospitals and PHCs",
    points:[
      "Staff dedicated heat-stroke corners with ORS, ice packs and cooling equipment.",
      "Report suspected heat-illness admissions daily to the state surveillance unit.",
      "Ensure uninterrupted power backup for cold chains and cooling units.",
      "Circulate the citizen advisory through ASHA and ANM field workers."
    ]
  },
  {
    audience:"Local Authorities",
    region:"North West India",
    severity:"heat",
    issued:"Today, 06:45 IST",
    headline:"Operationalise the municipal Heat Action Plan across identified hotspots",
    points:[
      "Open drinking-water kiosks and shaded rest areas at the four flagged hotspots.",
      "Extend public-park and cooling-centre timings through the afternoon window.",
      "Revise outdoor construction and municipal labour timings to before 11:00 IST.",
      "Broadcast the warning through public-address systems and local cable networks."
    ]
  }
];
