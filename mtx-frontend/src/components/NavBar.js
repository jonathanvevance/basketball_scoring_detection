import React, { useState } from 'react';
import { Divider } from '@material-ui/core';
import {
  Collapse,
  Navbar,
  NavbarToggler,
  NavbarBrand,
  NavLink,
  NavItem,
  Nav,
} from 'reactstrap';

const NavBar = (props) => {
  const [isOpen, setIsOpen] = useState(false);
  const toggle = () => setIsOpen(!isOpen);
  const list = () => {
    return [
      { path: '/results', name: 'RESULTS' },
      { path: '/visualization', name: 'VISUALIZATION' },
      { path: '/charts', name: 'CHARTS' },
      { path: '/videoscrub', name: 'VIDEO DATA' },
    ];
  };
  return (
    <div
      style={{
        fontFamily: 'Proxima Bold,sans-serif',
      }}
    >
      <Navbar style={{ backgroundColor: 'transparent' }} dark expand='md'>
        <NavbarBrand href='/' style={{ color: '#ee6730' }}>
          MTX-HackOlympics 2022
        </NavbarBrand>
        <NavbarToggler onClick={toggle} />
        <Collapse isOpen={isOpen} navbar>
          <Nav className='ms-auto' navbar>
            {list().map((data, key) => (
              <NavItem
                className='ml-auto'
                key={key}
                style={{ marginLeft: '10px' }}
              >
                <NavLink href={data.path} style={{ color: '#ee6730' }}>
                  {data.name}
                </NavLink>
              </NavItem>
            ))}
          </Nav>
        </Collapse>
      </Navbar>
      <Divider style={{ backgroundColor: 'black' }}></Divider>
    </div>
  );
};

export default NavBar;
