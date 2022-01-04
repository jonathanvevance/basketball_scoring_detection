import React from 'react';
import {
  Menu,
  MenuItem,
  Select,
  List,
  ListItem,
  ListItemText,
  Button,
} from '@material-ui/core';
import ChartControls from './chart';

const options = ['Ranges', 'Full', 'First 40', 'Last 60', 'Last 40', 'Last 20'];

const SimpleListMenu = (props) => {
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [selectedIndex, setSelectedIndex] = React.useState(1);
  const open = Boolean(anchorEl);
  const LineData = props.lineData;
  console.log(LineData);
  const [newData, setnewData] = React.useState(LineData);
  const handleClickListItem = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuItemClick = (event, index) => {
    setSelectedIndex(index);
    setAnchorEl(null);
    console.log(index);
    selectData(index);
    console.log(newData);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const selectData = (index) => {
    if (index === 1) setnewData(LineData);
    else if (index === 2) {
      var data = [];
      for (let i = 0; i < (LineData.length * 2) / 5; i++)
        data.push(LineData[i]);
      setnewData(data);
    }
  };

  return (
    <div>
      <List
        component='nav'
        aria-label='Chart settings'
        sx={{ bgcolor: 'background.paper' }}
      >
        <ListItem
          button
          id='lock-button'
          aria-haspopup='listbox'
          aria-controls='lock-menu'
          aria-label='Range Selector'
          aria-expanded={open ? 'true' : undefined}
          onClick={handleClickListItem}
        >
          <ListItemText
            primary='Select a Range'
            secondary={options[selectedIndex]}
          />
        </ListItem>
      </List>
      <Menu
        id='lock-menu'
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        MenuListProps={{
          'aria-labelledby': 'lock-button',
          role: 'listbox',
        }}
      >
        {options.map((option, index) => (
          <MenuItem
            key={option}
            disabled={index === 0}
            selected={index === selectedIndex}
            onClick={(event) => handleMenuItemClick(event, index)}
          >
            {option}
          </MenuItem>
        ))}
      </Menu>
      <div>{newData.length > 0 && <ChartControls data={newData} />}</div>
    </div>
  );
};

export default SimpleListMenu;
