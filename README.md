# Preline WebApp

A modern, fully functional web application built with HTML, CSS, JavaScript, and the Preline UI theme. This application features a complete dashboard with user authentication, admin panel, and responsive design.

## Features

### 🔐 Authentication System
- Secure login with email/password validation
- Session management (localStorage/sessionStorage)
- Password visibility toggle
- Remember me functionality
- Role-based access control (Admin/User)

### 🎨 Modern UI Components
- **Sidebar Navigation**: Responsive sidebar with smooth animations
- **Search Bar**: Real-time search functionality with dropdown results
- **Profile Widget**: User profile dropdown with account management
- **Dashboard**: Statistics cards with hover effects and gradient backgrounds
- **Admin Panel**: Comprehensive admin interface for user management

### 📱 Responsive Design
- Mobile-first approach
- Collapsible sidebar on mobile devices
- Touch-friendly interface
- Optimized for tablets and desktops

### 🛠 Admin Features
- User Management (Add, Edit, Delete users)
- Role assignment (Admin/User)
- Activity logs monitoring
- System settings configuration
- Security dashboard
- Real-time statistics

## Installation

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Live server for development (optional)

### Quick Start

1. **Clone or download the project files**
   ```bash
   git clone [repository-url]
   cd preline-webapp
   ```

2. **Install dependencies** (optional for development)
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```
   OR simply open `login.html` in your browser

### File Structure
```
preline-webapp/
├── login.html          # Login page
├── index.html          # Main dashboard
├── admin.html          # Admin panel
├── package.json        # Project dependencies
└── README.md          # This file
```

## Usage

### 1. Login
- Open `login.html` in your browser
- Use one of the following credentials:
  
  **Admin Account:**
  - Email: `admin@example.com`
  - Password: `admin123`
  
  **User Account:**
  - Email: `user@example.com`
  - Password: `user123`

### 2. Dashboard Features
After logging in, you'll have access to:

- **Navigation**: Use the sidebar to navigate between different sections
- **Search**: Global search functionality with real-time results
- **Profile**: Access your profile settings and logout
- **Statistics**: View real-time dashboard metrics
- **Quick Actions**: Perform common tasks directly from the dashboard

### 3. Admin Panel
If logged in as an admin, you can access the admin panel:

- **User Management**: Add, edit, or delete users
- **System Settings**: Configure application settings
- **Activity Logs**: Monitor user activities and system events
- **Security**: Review security status and login attempts

## Key Features Explained

### Authentication System
- **Secure Login**: Form validation with email format checking
- **Session Management**: Persistent login sessions with "Remember Me" option
- **Role-Based Access**: Different access levels for admin and regular users
- **Auto-Redirect**: Automatic redirection based on authentication status

### Dashboard Components
- **Responsive Sidebar**: Collapsible navigation with smooth animations
- **Search Functionality**: Real-time search with dropdown results
- **Profile Widget**: User information display with dropdown menu
- **Statistics Cards**: Interactive cards showing key metrics
- **Activity Feed**: Recent activities and quick actions

### Admin Panel Features
- **User Management**: Full CRUD operations for user accounts
- **Role Assignment**: Assign admin or user roles
- **Activity Monitoring**: Track user actions and system events
- **System Configuration**: Modify application settings
- **Security Dashboard**: Monitor security status and threats

## Customization

### Theming
The application uses the Preline UI framework with custom CSS:
- Modify colors in the CSS custom properties
- Adjust gradients and animations
- Customize component styles

### Adding New Features
1. **New Pages**: Create new HTML files and link them in the navigation
2. **API Integration**: Replace mock data with real API calls
3. **Database**: Connect to a backend database for persistent data
4. **Additional Components**: Add new dashboard widgets or admin features

## Browser Support
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Security Features
- Input validation and sanitization
- XSS protection
- CSRF protection considerations
- Secure session management
- Role-based access control

## Development

### Running Locally
1. Install a local server (like Live Server for VS Code)
2. Open the project in your development environment
3. Start the server and navigate to `login.html`

### Building for Production
1. Minify CSS and JavaScript files
2. Optimize images and assets
3. Configure proper server headers
4. Set up HTTPS

## Troubleshooting

### Common Issues
1. **Login Issues**: Check browser console for JavaScript errors
2. **Styling Problems**: Ensure Preline CSS is loading correctly
3. **Mobile Issues**: Check viewport meta tag and responsive styles
4. **Session Issues**: Clear browser storage and try again

### Browser Console
Open browser developer tools (F12) to check for:
- JavaScript errors
- Network request failures
- Console warnings

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License
This project is licensed under the MIT License.

## Support
For issues or questions:
1. Check the troubleshooting section
2. Review browser console for errors
3. Ensure all dependencies are loaded
4. Test in different browsers

## Future Enhancements
- Real-time notifications
- Advanced user roles and permissions
- Data export functionality
- Advanced analytics and reporting
- API integration for backend services
- Multi-language support
- Advanced security features

---

**Note**: This is a frontend-only application with simulated backend functionality. For production use, integrate with a proper backend service for authentication, data storage, and API endpoints. 