"""
Backward compatability module for the Microwave Uncertainty Framework (MUF).

The MUF saves csv files that are distributed across a folder system, with
an XML file that points to these files and describes their relation ships
to a nominal value. This module can parse that fodler structure, and save
data into that structure so that other uncertainty object schemas in
the package can read/write to them.
"""

import os  # to manipulate file paths
import xml.etree.ElementTree as ET  # to manipulate XML


def _indent(elem, level=0):
    """XML indenting function."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


class MUFMeasParser():
    """Object for reading legacy Microwave Uncertainry Framework xml files into uncertainty objects."""

    def __init__(self, file: str = None):
        """Initialize a MUFMeas parser.

        Parameters
        ----------
        file : str, optional
            Path to xml header file (usually has extension .meas), by default None.
        """
        super().__init__()

        self.etree = None
        self.parmameter_dict = None
        self.covariance_dict = None
        self.montecarlo_dict = None
        self.nominal_dict = None
        self.name = None
        self.file_ext = None

        if file is not None:
            self.open_meas(file)

    def open_meas(self, file: str):
        """Open up an xml header file and parse it for file paths and other info.

        Parameters
        ----------
        file : str
            Path to file to be opened.
        """
        self.etree = ET.parse(file)
        self.name = self._parse_name()
        self.parmameter_dict = self._parse_parameter_dict()
        self.covariance_dict = self._parse_covariance_dict()
        self.montecarlo_dict = self._parse_montecarlo_dict()
        self.nominal_dict = self._parse_nominal_dict()

    def open_data(self,
                  open_fcn: callable,
                  open_fcn_extra_args=(),
                  old_base_dir=None,
                  new_base_dir=None):
        """
        Loads data into an initialized object.

        Loads into memory all of the perturbed measurements in the sensitivity
        analysis, and all of the Monte-Carlo trials.

        You should call open_meas before opening data.
        """
        if old_base_dir is not None:
            # change nominal path
            self.nominal_dict["location"] = self.nominal_dict["location"].replace(old_base_dir, new_base_dir)

            # change montecarlo path
            for i, item in enumerate(self.montecarlo_dict):
                item["location"] = item["location"].replace(old_base_dir, new_base_dir)

            # change covariance path
            for i, item in enumerate(self.covariance_dict):
                item["location"] = item["location"].replace(old_base_dir, new_base_dir)

        # load nominal
        nominal_data = open_fcn(self.nominal_dict["location"], *open_fcn_extra_args)

        # load monte carlo
        montecarlo_data = []
        for i, item in enumerate(self.montecarlo_dict):
            montecarlo_data.append(open_fcn(item["location"], *open_fcn_extra_args))

        # load sensitivity analysis
        covariance_data = []
        umech_id = []
        for i, item in enumerate(self.covariance_dict):
            umech_id.append(self.covariance_dict[i]["parameter_location"])
            covariance_data.append(open_fcn(item["location"], *open_fcn_extra_args))

        self.init_from_data(self.name, montecarlo_data, nominal_data, covariance_data, umech_id)

    def init_from_data(self,
                       name: str,
                       montecarlo_data: list,
                       nominal_data: object,
                       covariance_data: list,
                       umech_id: list = None):
        """Generate an object from user supplied data.

        Parameters
        ----------
        name : _type_
            _description_
        montecarlo_data : list
            a list of data objects of the same type as
            nominal_data, representing Monte-Carlo trials. Can be empty.
        nominal_data : object
            a data object (usually a numpy array) representing
            a nominal value for some quantity
        covariance_data : list
            a list of data objects of the same type as
            nominal_data, representing the nominal data perturbed by
            various error mechanisms. Can be empty.
        umech_id : list, optional
            a list of strings with the
            same length as covariance_data. The locations of parameter files
            for error mechanisms. If empty, the location will be set to
        """
        # Erase old data to avoid confusion
        self.etree = None
        self.name = name
        self.parmameter_dict = None
        self.covariance_dict = None
        self.montecarlo_dict = None
        self.nominal_dict = None

        # monte-carlo
        self.montecarlo_data = montecarlo_data
        self.montecarlo_dict = []
        for i, data in enumerate(montecarlo_data):
            subdict = {"name": self.name + "_" + str(i),
                       "location": "unknown"}
            self.montecarlo_dict.append(subdict)

        # nominal
        self.nominal_dict = {"location": "",
                             "name": self.name}

        self.nominal_data = nominal_data

        # covariance
        self.covariance_data = covariance_data
        self.covariance_dict = []
        for i, data in enumerate(covariance_data):
            parameter_location = "unknown_" + str(i)
            if umech_id:
                parameter_location = umech_id[i]

            subdict = {"name": self.name + "_" + str(i),
                       "location": "unknown",
                       "parameter_location": parameter_location}
            self.covariance_dict.append(subdict)

    def save_data(self,
                  target_dir: str,
                  save_fcn: callable = None,
                  save_fcn_extra_args=(),
                  file_ext: str = None):
        """Save data to disk, call before save_meas.

        Given a target directory, creates a "<self.name>_Support" folder with
        subdirectories "Covariance", containing files consisting of perturbed measurements
        for the sensitivity analysis, and "MonteCarlo", containing files consisting
        of Monte-Carlo trials. The save_fcn provide is called as
        save_fcn_extra_args(data,filepath,*save_fcn_extra_args).

        Parameters
        ----------
        target_dir : str
            directory where data should be saved
        save_fcn : callable, optional
            function that saves data in the propper format. Should take
            the data to be saved as the first argument and the file path as
            the second.
        save_fcn_extra_args : tuple, optional
           extra arguments for the save function if necessary, by default ().
        file_ext : str, optional
            Extension to use for filepaths, by default None
        """
        self.file_ext = file_ext
        if self.file_ext is None:
            self.file_ext = ".complex"

        out_dir = target_dir + "//" + self.name + "_Support//"
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        if self.montecarlo_dict and not os.path.isdir(out_dir + "MonteCarlo//"):
            os.mkdir(out_dir + "MonteCarlo//")

        if self.covariance_dict and not os.path.isdir(out_dir + "Covariance//"):
            os.mkdir(out_dir + "Covariance//")

        # save montecarlo
        for i, item in enumerate(self.montecarlo_dict):
            filename = out_dir + "MonteCarlo//" + self.montecarlo_dict[i]['name'] + self.file_ext
            self.montecarlo_dict[i]['location'] = filename
            save_fcn(self.montecarlo_data[i], filename, *save_fcn_extra_args)

        # save nominal
        filename = out_dir + self.nominal_dict['name'] + self.file_ext
        self.nominal_dict['location'] = filename
        save_fcn(self.nominal_data, filename, *save_fcn_extra_args)

        # save covariance
        for i, item in enumerate(self.covariance_dict):
            filename = out_dir + "Covariance//" + self.covariance_dict[i]['name'] + self.file_ext
            self.covariance_dict[i]['location'] = filename
            save_fcn(self.covariance_data[i], filename, *save_fcn_extra_args)

    def _parse_parameter_dict(self):
        """Get a dictionary of 'name':'parameter_name' pairs to correlate covariance files with their mechanism names."""
        out_dictionary = {}
        names = [x.attrib["Text"] for x in self.etree.findall(".//PerturbedSParams/Item/SubItem[@Index='0']")]
        mechanisms = [x.attrib["Text"] for x in self.etree.findall(".//PerturbedSParams/Item/SubItem[@Index='2']")]
        for index, name in enumerate(names):
            split_parameter_name = os.path.split(mechanisms[index])[-1]
            parameter_name = split_parameter_name.split(".")[0]
            out_dictionary[name] = parameter_name

        return out_dictionary

    def _parse_covariance_dict(self):
        """Get a list of dictionaries that has the keys name, location, and parameter_location."""
        covariance_list = []
        names = [x.attrib["Text"] for x in self.etree.findall(".//PerturbedSParams/Item/SubItem[@Index='0']")]
        locations = [x.attrib["Text"] for x in self.etree.findall(".//PerturbedSParams/Item/SubItem[@Index='1']")]
        mechanisms = [x.attrib["Text"] for x in self.etree.findall(".//PerturbedSParams/Item/SubItem[@Index='2']")]
        for index, name in enumerate(names):
            name_location_dictionary = {"name": name, "location": locations[index], "parameter_location": mechanisms[index]}
            covariance_list.append(name_location_dictionary)

        return covariance_list

    def _parse_montecarlo_dict(self):
        """Get a list of dictionaries that has the keys name, location."""
        montecarlo_list = []
        names = [x.attrib["Text"] for x in self.etree.findall(".//MonteCarloPerturbedSParams/Item/SubItem[@Index='0']")]
        locations = [x.attrib["Text"] for x in self.etree.findall(".//MonteCarloPerturbedSParams/Item/SubItem[@Index='1']")]
        for index, name in enumerate(names):
            name_location_dictionary = {"name": name, "location": locations[index]}
            montecarlo_list.append(name_location_dictionary)

        return montecarlo_list

    def _parse_name(self):
        """Get the measurement name."""
        return self.etree.find("Controls/MeasurementName").attrib['ControlText']

    def _parse_nominal_dict(self):
        """Return a single dictionary with nominal name and location."""
        nominal_dictionary = {}
        location = list(map(lambda x: x.attrib["Text"],
                            self.etree.findall(".//MeasSParams/Item/SubItem[@Index='1']")))[0]

        name = os.path.split(location)[-1].split(".")[0]
        nominal_dictionary["location"] = location
        nominal_dictionary["name"] = name
        return nominal_dictionary

    def save_meas(self, output_file):
        """Save the xml header.

        Saves a MUF-style .meas file. This function only writes the XML file,
        it does not save the raw data.

        If you are also saving raw data, you should do that first, because
        that function will alter file paths.

        Parameters
        ----------
        output_file : str
            Path to the xml header.
        """
        # **********************************************************************
        # These attributes document the provenance of the file
        # For now, I'm not worryting about them
        # **********************************************************************
        CorrectedMeasurement_attrib = {"FileName": output_file,
                                       "CreationTime": "unknown",
                                       "AssemblyVersion": "unknown",
                                       "UserName": "unknown"}

        CreatedByUserName_attrbib = {"Enabled": "True", "Text": "unknown"}
        CreatedByProgram_attrib = {"Enabled": "True", "Text": "RockyMountainAdvanced"}
        CreatedByVersion_attrib = {"Enabled": "True", "Text": "uknonwn"}
        CreatedByDate_attrib = {"Enabled": "True", "Text": "unknown"}
        CreatedByFirstFileName_attrib = {"Enabled": "True", "Text": "unknown"}
        CreatedByUpdatedFileName_attrib = {"Enabled": "True", "Text": "unknown"}
        CreatedFromFirstFileName_attrib = {"Enabled": "True", "Text": "unknown"}
        CreatedFromUpdatedFileName_attrib = {"Enabled": "True", "Text": "unknown"}
        MyVersion_attrib = {"Enabled": "True", "Text": "unknown"}

        MeasurementName_attrib = {"ControlType": "System.Windows.Forms.TextBox",
                                  "ControlText": self.name,
                                  "FullName": "Me_SplitContainer2__MeasurementName",
                                  "Enabled": "True",
                                  "Visible": "True"}

        MeasurementDocumentation_attrib = {"ControlType": "System.Windows.Forms.TextBox",
                                           "ControlText": "unknown",
                                           "FullName": "Me_SplitContainer2__GroupBox5_MeasurementDocumentation"}

        CorrectedMeasurement = ET.Element("CorrectedMeasurement", CorrectedMeasurement_attrib)
        MenuStripTextBoxes = ET.SubElement(CorrectedMeasurement, "MenuStripTextBoxes")
        ET.SubElement(MenuStripTextBoxes, "CreatedByUserName", CreatedByUserName_attrbib)
        ET.SubElement(MenuStripTextBoxes, "CreatedByProgram", CreatedByProgram_attrib)
        ET.SubElement(MenuStripTextBoxes, "CreatedByVersion", CreatedByVersion_attrib)
        ET.SubElement(MenuStripTextBoxes, "CreatedByDate", CreatedByDate_attrib)
        ET.SubElement(MenuStripTextBoxes, "CreatedByFirstFileName", CreatedByFirstFileName_attrib)
        ET.SubElement(MenuStripTextBoxes, "CreatedByUpdatedFileName", CreatedByUpdatedFileName_attrib)
        ET.SubElement(MenuStripTextBoxes, "CreatedFromFirstFileName", CreatedFromFirstFileName_attrib)
        ET.SubElement(MenuStripTextBoxes, "CreatedFromUpdatedFileName", CreatedFromUpdatedFileName_attrib)
        ET.SubElement(MenuStripTextBoxes, "MyVersion", MyVersion_attrib)

        # **********************************************************************
        # Nominal, Covariance, and Monte-Carlo information is under Controls
        # **********************************************************************
        Controls = ET.SubElement(CorrectedMeasurement, "Controls")
        ET.SubElement(Controls, "MeasurementName", MeasurementName_attrib)
        ET.SubElement(Controls, "MeasurementDocumentation", MeasurementDocumentation_attrib)

        # **********************************************************************
        # Monte-Carlo
        # **********************************************************************
        MonteCarloPerturbedSParams_attrib = {"ControlType": "CustomFormControls.FLV_VariableDetailsList",
                                             "FullName": "Me_SplitContainer2__GroupBox3_Panel2_MonteCarloPerturbedSParams",
                                             "Count": str(len(self.montecarlo_dict))}

        MonteCarloPerturbedSParams = ET.SubElement(Controls, "MonteCarloPerturbedSParams", MonteCarloPerturbedSParams_attrib)
        for i in range(len(self.montecarlo_dict)):
            Item_attrib = {"Index": str(i),
                           "Text": self.montecarlo_dict[i]["name"],
                           "Count": "2"}

            SubItem0_attrib = {"Index": "0",
                               "Text": self.montecarlo_dict[i]["name"]}

            SubItem1_attrib = {"Index": "1",
                               "Text": self.montecarlo_dict[i]["location"]}

            Item = ET.SubElement(MonteCarloPerturbedSParams, "Item", Item_attrib)
            ET.SubElement(Item, "SubItem", SubItem0_attrib)
            ET.SubElement(Item, "SubItem", SubItem1_attrib)

        # **********************************************************************
        # Nominal
        # **********************************************************************
        MeasSParams_attrib = {"ControlType": "CustomFormControls.FLV_FixedDetailsList",
                              "FullName": "Me_SplitContainer2__GroupBox2_Panel3_MeasSParams",
                              "Count": "1"}

        MeasSParams = ET.SubElement(Controls, "MeasSParams", MeasSParams_attrib)
        Item_attrib = {"Index": "0",
                       "Text": self.nominal_dict["name"],
                       "Count": "2"}

        SubItem0_attrib = {"Index": "0",
                           "Text": self.nominal_dict["name"]}

        SubItem1_attrib = {"Index": "1",
                           "Text": self.nominal_dict["location"]}

        Item = ET.SubElement(MeasSParams, "Item", Item_attrib)
        ET.SubElement(Item, "SubItem", SubItem0_attrib)
        ET.SubElement(Item, "SubItem", SubItem1_attrib)

        # **********************************************************************
        # Covariance
        # **********************************************************************
        PerturbedSParams_attrib = {"ControlType": "CustomFormControls.FLV_VariableDetailsListMeas",
                                   "FullName": "Me_SplitContainer2__GroupBox1_Panel1_PerturbedSParams",
                                   "Count": str(len(self.covariance_dict))}

        PerturbedSParams = ET.SubElement(Controls, "PerturbedSParams", PerturbedSParams_attrib)
        for i in range(len(self.covariance_dict)):
            Item_attrib = {"Index": str(i),
                           "Text": self.covariance_dict[i]["name"],
                           "Count": "3"}

            SubItem0_attrib = {"Index": "0",
                               "Text": self.covariance_dict[i]["name"]}

            SubItem1_attrib = {"Index": "1",
                               "Text": self.covariance_dict[i]["location"]}

            SubItem2_attrib = {"Index": "2",
                               "Text": self.covariance_dict[i]["parameter_location"]}

            Item = ET.SubElement(PerturbedSParams, "Item", Item_attrib)
            ET.SubElement(Item, "SubItem", SubItem0_attrib)
            ET.SubElement(Item, "SubItem", SubItem1_attrib)
            ET.SubElement(Item, "SubItem", SubItem2_attrib)

        # **********************************************************************
        # The end
        # **********************************************************************

        _indent(CorrectedMeasurement, level=1)
        new_ET = ET.ElementTree(CorrectedMeasurement)
        new_ET.write(output_file, xml_declaration=True)
